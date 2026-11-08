"""Daily event-loop backtest (monthly decisions, next-open fills, drifting weights).

Semantics (fixed before any result was read):
* decision at the close of the last session of month t; fill at the OPEN of the
  next session; returns accrue open-to-open, so the fill price is the label's
  entry price (label = open[t+22]/open[t+1]-1).
* constraints (long-only, fully invested, max weight) are enforced on the target
  weights inside the loop; a name with no valid open on the fill day is not
  traded (keeps its drifted weight).
* per-leg cost = (commission + half-spread) / raw open of that name on that day,
  times |weight change|; deducted from that day's return.
* valuation of a held name on a missing bar uses its last valid open (marking
  only, never trading).
* the vol-target overlay scales gross exposure with the strategy's own realised
  volatility up to the decision close (t-1 relative to the fill), rest in cash.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .data import month_end_sessions, per_leg_cost_bps, wide


@dataclass
class Market:
    open_adj: pd.DataFrame       # for returns
    close_adj: pd.DataFrame      # for allocator risk estimates
    cost: pd.DataFrame           # per-leg cost fraction at the raw open (commission + half-spread)
    sessions: pd.DatetimeIndex
    spread: pd.DataFrame = None  # half-spread part of `cost` (for spread-only stress)
    oo_ret: pd.DataFrame = field(init=False)   # return from open d to open d+1, indexed by d

    def __post_init__(self):
        o = self.open_adj.ffill()                 # marking only
        self.oo_ret = (o.shift(-1) / o - 1).fillna(0.0)


def market_from_panel(df: pd.DataFrame) -> Market:
    from .data import half_spread_usd
    o, c = wide(df, "open"), wide(df, "close")
    open_raw = wide(df, "open_raw").reindex(o.index)
    spread = pd.Series({s: half_spread_usd(s) for s in open_raw.columns}) / open_raw
    return Market(o, c, per_leg_cost_bps(df).reindex(o.index), pd.DatetimeIndex(o.index), spread)


# ----------------------------------------------------------------------------- allocators
def cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """Water-filling: clip at cap and redistribute the excess pro rata; if every
    name is capped the remainder stays in cash."""
    w = w.clip(lower=0)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w.clip(upper=cap)


def inverse_vol(names, close: pd.DataFrame, t, window=63):
    r = np.log(close.loc[:t, names]).diff().iloc[-window:]
    iv = 1 / r.std().replace(0, np.nan)
    return (iv / iv.sum()).fillna(0)


def hrp(names, close: pd.DataFrame, t, window=63):
    r = np.log(close.loc[:t, names]).diff().iloc[-window:].dropna(axis=1, how="any")
    if r.shape[1] < 2:
        return pd.Series(1.0 / len(names), index=names)
    cov, corr = r.cov(), r.corr()
    dist = np.sqrt(((1 - corr) / 2).clip(lower=0))
    order = list(corr.index[leaves_list(linkage(squareform(dist.values, checks=False), "single"))])
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            a, b = clusters[i], clusters[i + 1]
            va = _cluster_var(cov, a); vb = _cluster_var(cov, b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha; w[b] *= 1 - alpha
    return w.reindex(names).fillna(0)


def _cluster_var(cov, items):
    c = cov.loc[items, items]
    ivp = 1 / np.diag(c); ivp /= ivp.sum()
    return float(ivp @ c.values @ ivp)


# ----------------------------------------------------------------------------- engine
def decision_dates(sessions: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    me = month_end_sessions(sessions)
    # the decision whose fill is the first session of the window is the prior month-end
    return me[(me >= pd.Timestamp(start) - pd.offsets.MonthEnd(1) - pd.Timedelta(days=3)) & (me < pd.Timestamp(end))]


def run(scores: pd.DataFrame | None, mkt: Market, start, end, top_n: int = 10, allocator: str = "equal_weight",
        cost_mult: float = 1.0, max_weight: float = 0.10, vol_target: float | None = None, vol_window: int = 63,
        fixed_weights: dict | None = None, universe_mask: pd.DataFrame | None = None, flat_cost: float | None = None,
        spread_mult: float = 1.0) -> dict:
    """scores: date x symbol on decision dates (higher = buy). fixed_weights: static benchmark weights.
    universe_mask (date x symbol bool) with scores=None and fixed_weights=None -> equal weight all eligible."""
    S = mkt.sessions
    days = S[(S >= pd.Timestamp(start)) & (S <= pd.Timestamp(end))]
    dec = decision_dates(S, start, end)
    exec_map = {}
    for t in dec:
        i = S.get_loc(t)
        if i + 1 < len(S) and S[i + 1] in days:
            exec_map[S[i + 1]] = t
    syms = mkt.open_adj.columns
    w = pd.Series(0.0, index=syms)
    out_ret, out_gross, out_turn, out_exp, holdings = {}, {}, {}, {}, {}
    raw_hist = []                                     # unscaled strategy returns for the vol overlay
    scale = 1.0
    for d in days:
        cost_today = 0.0
        if d in exec_map:
            t = exec_map[d]
            if fixed_weights is not None:
                target = pd.Series(fixed_weights).reindex(syms).fillna(0.0)
            elif scores is None:
                m = universe_mask.loc[t].reindex(syms).fillna(False)
                target = (m / m.sum()).astype(float)
            else:
                sc = scores.loc[t].dropna() if t in scores.index else pd.Series(dtype=float)
                names = sc.sort_values(ascending=False).index[:top_n].tolist()
                if allocator == "equal_weight":
                    raw = pd.Series(1.0 / top_n, index=names)        # unfilled slots -> cash
                elif allocator == "inverse_vol":
                    raw = inverse_vol(names, mkt.close_adj, t)
                elif allocator == "hrp":
                    raw = hrp(names, mkt.close_adj, t)
                else:
                    raise ValueError(allocator)
                raw = cap_weights(raw, max_weight) if allocator != "equal_weight" else raw
                target = raw.reindex(syms).fillna(0.0)
            if vol_target is not None:
                hist = pd.Series(raw_hist[:-1][-vol_window:])   # drop open(t)->open(t+1): not known at the close of t
                rv = hist.std() * np.sqrt(252) if len(hist) >= 21 else np.nan
                scale = min(1.0, vol_target / rv) if np.isfinite(rv) and rv > 0 else 1.0
                target = target * scale
            assert target.min() >= -1e-12 and target.sum() <= 1 + 1e-9
            tradable = mkt.open_adj.loc[d].notna()
            stuck = ~tradable & (w > 0)
            target = target.where(tradable, w)              # no trade on a missing bar
            if target.sum() > 1 + 1e-12:                    # keep gross <= 1: scale the tradable names down
                free = target[~stuck].sum(); room = 1 - target[stuck].sum()
                target[~stuck] *= max(room, 0) / free if free > 0 else 0
            dw = (target - w).abs()
            if flat_cost is not None:
                leg = pd.Series(flat_cost, index=syms)
            else:
                leg = mkt.cost.loc[d].reindex(syms).fillna(0.0)
                if spread_mult != 1.0:
                    leg = leg + (spread_mult - 1.0) * mkt.spread.loc[d].reindex(syms).fillna(0.0)
            cost_today = float((dw * leg).sum() * cost_mult)
            out_turn[d] = float(dw.sum())
            w = target
            holdings[d] = w[w > 0].round(4).to_dict()
        r = mkt.oo_ret.loc[d].reindex(syms).fillna(0.0)
        gross = float((w * r).sum())
        out_gross[d] = gross
        out_ret[d] = gross - cost_today
        out_exp[d] = float(w.sum())
        raw_hist.append(gross / scale if scale > 0 else gross)
        w = w * (1 + r) / (1 + gross) if (1 + gross) != 0 else w
    # out_ret[d] is the open(d)->open(d+1) return; index by d+1 so that it is realised that day
    # the last session has no next open inside the data: its return is unknown -> dropped (no phantom day)
    keep = [x for x in out_ret if S.get_loc(x) + 1 < len(S)]
    nxt = pd.DatetimeIndex([S[S.get_loc(x) + 1] for x in keep])
    def ser(dct): return pd.Series([dct[x] for x in keep], index=nxt)
    return {"ret": ser(out_ret), "gross": ser(out_gross), "turnover": pd.Series(out_turn), "exposure": ser(out_exp), "holdings": holdings}


def annual_turnover(res: dict) -> float:
    yrs = len(res["ret"]) / 252
    return float(res["turnover"].sum() / yrs)


def breakeven_cost(res_gross_zero_cost: dict) -> float:
    """c* (per leg, fraction) = annualised gross return / annual two-way turnover."""
    r = res_gross_zero_cost["gross"]
    return float(r.mean() * 252 / annual_turnover(res_gross_zero_cost))
