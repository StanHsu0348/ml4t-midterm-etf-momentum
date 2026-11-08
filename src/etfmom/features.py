"""Labels and candidate features. Every value at date t uses data with date <= t.

The feature register (REGISTRY) is the Table-8.1-style hypothesis table: one row
per candidate with role, driver hypothesis, lookback, representation, known
failure mode and the sign the mechanism predicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import eligibility, wide

ANN = np.sqrt(252)

REGISTRY = pd.DataFrame([
    # name, role, driver hypothesis, lookback, representation, failure mode, expected sign
    ("mom_252_21", "signal", "Persistence: under-reaction / slow-moving capital (12-1 momentum)", 252, "return, XS rank", "momentum crashes after volatility spikes", +1),
    ("mom_126_21", "signal", "Persistence at 6-1 months", 126, "return, XS rank", "regime concentration in one asset class", +1),
    ("mom_63", "signal", "Persistence at 3 months (faster trend)", 63, "return, XS rank", "whipsaw in range-bound markets", +1),
    ("volscaled_mom_252_21", "signal", "Persistence per unit risk (TS-momentum scaling)", 252, "return / vol, XS rank", "low-vol bond ETFs dominate the ranking", +1),
    ("sharpe_126", "signal", "Trend earned with less dispersion repeats", 126, "mean/std of log returns, XS rank", "unbounded when vol -> 0", +1),
    ("dist_52w_high", "signal", "Anchoring on the 52-week high delays reaction to news", 252, "close / 252d max - 1, XS rank", "piles up at 0 in sustained bull runs", +1),
    ("sma_ratio_200", "signal", "Trend-following: price above its 200d average", 200, "close / SMA200 - 1, XS rank", "lags turning points by months", +1),
    ("trend_consistency_63", "signal", "Frog-in-the-pan: gradual information diffuses slowly", 63, "share of up sessions, XS rank", "noisy for low-vol funds", +1),
    ("mom_accel", "signal", "Trend acceleration: recent leg vs older leg", 126, "mom_63 - (mom_126_21 - mom_63), XS rank", "flags reversals as acceleration", +1),
    ("rev_21", "signal", "Short-term reversal: liquidity provision to 1-month flow pressure", 21, "return, XS rank", "ETF reversal weaker than single stocks", -1),
    ("rev_5", "signal", "Weekly reversal: liquidity provision", 5, "return, XS rank", "dominated by noise at monthly horizon", -1),
    ("vol_63", "signal", "Low-risk anomaly: leverage-constrained investors overpay for risk", 63, "annualised std, XS rank", "mixes asset classes with structurally different vol", -1),
    ("beta_spy_126", "signal", "Betting-against-beta: funding constraints", 126, "OLS beta to SPY, XS rank", "beta unstable across regimes", -1),
    ("max_ret_21", "signal", "Lottery demand: investors overpay for extreme upside", 21, "max daily return, XS rank", "driven by single events", -1),
    ("skew_63", "signal", "Skewness preference: positively skewed assets overpriced", 63, "sample skewness, XS rank", "estimation noise in 63 obs", -1),
    ("amihud_63", "signal", "Illiquidity premium: compensation for price impact", 63, "mean |r| / $volume, XS rank", "ETF liquidity != underlying liquidity", +1),
    ("volume_trend_21_126", "signal", "Attention: rising participation confirms the move", 126, "log(ADV21/ADV126), XS rank", "index events and rebalancing days", +1),
    ("asset_rvol_pct", "state", "Own volatility vs own history: stress state of the asset", 252, "percentile of vol_21 in own 252d", "single shock dominates", 0),
    ("mkt_rvol_pct", "state", "Market risk environment (SPY 21d vol vs expanding history)", 252, "expanding percentile", "level shifts in vol regime", 0),
    ("mkt_trend", "state", "Market trend state (SPY vs SMA200)", 200, "ratio", "late at turning points", 0),
    ("corr_spy_tlt_63", "state", "Hedge regime: stock-bond correlation", 63, "level", "nonsynchronous trading, one pair stands for all", 0),
    ("xs_disp_21", "state", "Opportunity set: cross-sectional dispersion of 1-month returns", 21, "XS std", "dominated by commodity outliers", 0),
], columns=["feature", "role", "hypothesis", "lookback", "representation", "failure_mode", "expected_sign"])

SIGNALS = REGISTRY.query("role == 'signal'").feature.tolist()
STATES = REGISTRY.query("role == 'state'").feature.tolist()


def _xs_rank(w: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank centred on 0 ([-0.5, 0.5]) among eligible names."""
    w = w.where(mask)
    return w.rank(axis=1, pct=True) - 0.5


def compute_raw_features(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    close, high = wide(df, "close"), wide(df, "high")
    vol_raw = wide(df, "volume_raw")
    dollar = wide(df, "close_raw") * vol_raw
    r = np.log(close).diff()
    rs = close.pct_change(fill_method=None)
    f: dict[str, pd.DataFrame] = {}
    f["mom_252_21"] = close.shift(21) / close.shift(252) - 1
    f["mom_126_21"] = close.shift(21) / close.shift(126) - 1
    f["mom_63"] = close / close.shift(63) - 1
    vol252 = r.rolling(252, min_periods=200).std() * ANN
    f["volscaled_mom_252_21"] = f["mom_252_21"] / vol252
    f["sharpe_126"] = r.rolling(126, min_periods=100).mean() / r.rolling(126, min_periods=100).std() * ANN
    f["dist_52w_high"] = close / close.rolling(252, min_periods=200).max() - 1
    f["sma_ratio_200"] = close / close.rolling(200, min_periods=180).mean() - 1
    f["trend_consistency_63"] = (r > 0).astype(float).where(r.notna()).rolling(63, min_periods=50).mean()
    f["mom_accel"] = f["mom_63"] - (f["mom_126_21"] - f["mom_63"])
    f["rev_21"] = close / close.shift(21) - 1
    f["rev_5"] = close / close.shift(5) - 1
    f["vol_63"] = r.rolling(63, min_periods=50).std() * ANN
    spy = r["SPY"]
    cov = r.rolling(126, min_periods=100).cov(spy)
    f["beta_spy_126"] = cov.div(spy.rolling(126, min_periods=100).var(), axis=0)
    f["max_ret_21"] = rs.rolling(21, min_periods=17).max()
    f["skew_63"] = r.rolling(63, min_periods=50).skew()
    f["amihud_63"] = np.log((rs.abs() / dollar).rolling(63, min_periods=50).mean() * 1e9)
    f["volume_trend_21_126"] = np.log(dollar.rolling(21, min_periods=17).mean() / dollar.rolling(126, min_periods=100).mean())
    vol21 = r.rolling(21, min_periods=17).std() * ANN
    f["asset_rvol_pct"] = vol21.rolling(252, min_periods=200).rank(pct=True)
    # date-level states (same value for every symbol on a date)
    spy_vol = vol21["SPY"]
    mk = pd.DataFrame(index=close.index)
    mk["mkt_rvol_pct"] = spy_vol.expanding(min_periods=252).rank(pct=True)
    mk["mkt_trend"] = close["SPY"] / close["SPY"].rolling(200, min_periods=180).mean() - 1
    mk["corr_spy_tlt_63"] = r["SPY"].rolling(63, min_periods=50).corr(r["TLT"])
    f["_mkt"] = mk
    f["_ret21"] = f["rev_21"]
    return f


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Long panel (date, symbol): raw + XS-ranked features, state variables, labels, eligibility."""
    f = compute_raw_features(df)
    elig = eligibility(df)
    elig = elig.reindex(index=f["mom_252_21"].index, columns=f["mom_252_21"].columns).fillna(False).astype(bool)
    mk = f.pop("_mkt")
    ret21 = f.pop("_ret21")
    mk["xs_disp_21"] = ret21.where(elig).std(axis=1)
    open_ = wide(df, "open").reindex(elig.index)
    close_ = wide(df, "close").reindex(elig.index)
    labels = {
        "fwd_open_ret_21d": open_.shift(-22) / open_.shift(-1) - 1,
        "fwd_open_ret_5d": open_.shift(-6) / open_.shift(-1) - 1,
        "fwd_open_ret_63d": open_.shift(-64) / open_.shift(-1) - 1,
        "fwd_close_ret_21d": close_.shift(-21) / close_ - 1,      # for the label-convention comparison only
    }
    cols = {}
    for k, v in f.items():
        v = v.replace([np.inf, -np.inf], np.nan)
        cols[k + "__raw"] = v.where(elig)
        cols[k] = v.where(elig) if k in STATES else _xs_rank(v, elig & v.notna())
    for k, v in labels.items():
        cols[k] = v.where(elig)
    stacked = {k: v.stack(future_stack=True) for k, v in cols.items()}
    panel = pd.DataFrame(stacked)
    panel.index.names = ["date", "symbol"]
    panel["eligible"] = elig.stack(future_stack=True)
    panel = panel[panel.eligible].drop(columns="eligible")
    panel = panel.join(mk, on="date")
    return panel.sort_index()
