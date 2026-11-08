# %% [markdown]
# # 02 · Data: sources, point-in-time pipeline, quality report
# Development sample only (holdout rows are removed by `load_panel`). Coverage
# statistics for the holdout years are *not* computed here.

# %%
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etfmom.config import FIG, TAB, ROOT, dev_end
from etfmom.data import (ACTIONS_PATH, META_PATH, eligibility, load_panel, month_end_sessions,
                         per_leg_cost_bps, universe, wide)

plt.style.use("seaborn-v0_8-whitegrid")
meta = json.loads(META_PATH.read_text())
df = load_panel(raw=True)   # raw: quality checks run before cleaning
print(meta)
print("dev rows:", len(df), "| last dev date:", df.date.max().date(), "| dev_end:", dev_end().date())
assert df.date.max() <= dev_end()

# %% [markdown]
# ## Duplicates, OHLC invariants, non-positive prices

# %%
dup = int(df.duplicated(["symbol", "date"]).sum())
tol = 1e-6
inv = pd.DataFrame({
    "high<low": df.high < df.low * (1 - tol),
    "open>high": df.open > df.high * (1 + tol),
    "open<low": df.open < df.low * (1 - tol),
    "close>high": df.close > df.high * (1 + tol),
    "close<low": df.close < df.low * (1 - tol),
    "price<=0": (df[["open", "high", "low", "close"]] <= 0).any(axis=1),
    "volume==0": df.volume == 0,
    "nan_ohlc": df[["open", "high", "low", "close"]].isna().any(axis=1),
})
quality = inv.sum().rename("violations").to_frame()
quality.loc["duplicate (symbol,date)"] = dup
quality["share_%"] = (quality.violations / len(df) * 100).round(4)
print(quality)
bad_rows = inv.any(axis=1)
by_sym = df[bad_rows].assign(**{k: inv.loc[bad_rows, k] for k in inv}).groupby("symbol")[list(inv)].sum()
by_sym = by_sym[by_sym.sum(axis=1) > 0].sort_values("volume==0", ascending=False)
print(by_sym.head(10))

# %% [markdown]
# Handling rule (no silent forward-fill): a row with an OHLC violation, a
# non-positive price, or zero volume is **flagged and excluded** — its open is
# set to NaN so no trade can be filled on it, and its return is NaN so it cannot
# enter a feature or label.

# %%
flag = inv[["high<low", "open>high", "open<low", "close>high", "close<low", "price<=0", "volume==0", "nan_ohlc"]].any(axis=1)
print(f"flagged rows excluded: {int(flag.sum())} ({flag.mean()*100:.3f}%)")

# %% [markdown]
# ## Missing sessions per symbol (vs. the NYSE session calendar implied by SPY)

# %%
cal = pd.DatetimeIndex(sorted(df.loc[df.symbol == "SPY", "date"]))
close = wide(df, "close").reindex(cal)
first = close.apply(lambda s: s.first_valid_index())
gaps = {}
for sym in close.columns:
    s = close.loc[first[sym]:, sym]
    gaps[sym] = int(s.isna().sum())
gaps = pd.Series(gaps).sort_values(ascending=False)
print("symbols with interior missing sessions:", int((gaps > 0).sum()), "| total missing:", int(gaps.sum()))
print(gaps.head(8))

# %% [markdown]
# ## Coverage over time: eligible names per month-end decision date

# %%
elig = eligibility(df).reindex(cal)
me = month_end_sessions(cal)
cov = elig.loc[me].sum(axis=1)
listed = close.loc[me].notna().sum(axis=1)
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(listed.index, listed, label="listed (has price)", color="0.6")
ax.plot(cov.index, cov, label="eligible (history >= 273d & median $ADV63 >= $10M)", color="C0")
ax.set_title("Q: how many names can the strategy actually trade each month?")
ax.set_ylabel("# ETFs"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "02_coverage.png", dpi=150)
cov.describe()

# %% [markdown]
# ## Corporate-action check: raw vs adjusted prices around a split

# %%
act = pd.read_parquet(ACTIONS_PATH)
act["date"] = pd.to_datetime(act["date"])
splits = act[(act.kind == "split") & (act.date <= dev_end())].sort_values("date")
print(splits)
sym, sdate = splits.iloc[-1][["symbol", "date"]] if len(splits) else ("SPY", pd.Timestamp("2020-01-02"))
w = df[(df.symbol == sym) & df.date.between(sdate - pd.Timedelta(days=40), sdate + pd.Timedelta(days=40))]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(w.date, w.close_raw, label="traded (unadjusted) close"); ax[0].plot(w.date, w.close, label="adjusted close")
ax[0].axvline(sdate, color="k", ls=":"); ax[0].set_title(f"{sym}: split ex-date {sdate.date()}"); ax[0].legend()
spy = df[(df.symbol == "SPY") & (df.date.dt.year == 2019)]
ax[1].plot(spy.date, spy.close_raw / spy.close_raw.iloc[0], label="raw (price only)")
ax[1].plot(spy.date, spy.close / spy.close.iloc[0], label="adjusted (total return)")
for d in act[(act.symbol == "SPY") & (act.kind == "dividend") & (act.date.dt.year == 2019)].date:
    ax[1].axvline(d, color="0.7", lw=0.8)
ax[1].set_title("SPY 2019: dividends (grey) separate raw from adjusted"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "02_corporate_actions.png", dpi=150)
r_raw = w.set_index("date").close_raw.pct_change(); r_adj = w.set_index("date").close.pct_change()
print(f"{sym} unadjusted return on ex-date: {r_raw.loc[sdate]:+.2%} | adjusted: {r_adj.loc[sdate]:+.2%}")
print("dollar volume invariant to the rebuild:", np.allclose((w.close_raw*w.volume_raw), (w.close_raw/w.split_factor*w.volume)))

# %% [markdown]
# ## Survivorship: names that left the sample

# %%
last = close.apply(lambda s: s.last_valid_index())
left = last[last < cal[-1] - pd.Timedelta(days=10)]
print("names whose price series ends before the dev sample end:", len(left))
lst = pd.DataFrame({"first_date": first, "last_date": last}).join(universe().set_index("symbol"))
lst.to_csv(TAB / "02_listing_dates.csv")
lst.sort_values("first_date").tail(8)

# %% [markdown]
# ## Independent vendor snapshot cross-check (textbook parquet, if present)

# %%
from pathlib import Path
tb = ROOT.parent / "machine-learning-for-trading-main" / "data" / "etfs" / "market" / "etf_universe.parquet"
xcheck = None
if tb.exists():
    t = pd.read_parquet(tb, columns=["timestamp", "symbol", "close"])
    t["date"] = pd.to_datetime(t.timestamp).dt.normalize()
    rt = t.pivot(index="date", columns="symbol", values="close").pct_change()
    ro = close.pct_change()
    common = rt.index.intersection(ro.index); syms = rt.columns.intersection(ro.columns)
    diff = (rt.loc[common, syms] - ro.loc[common, syms]).abs()
    xcheck = {"symbols": len(syms), "sessions": len(common),
              "share_abs_diff_gt_10bp": float((diff > 0.001).sum().sum() / diff.notna().sum().sum()),
              "median_abs_diff_bp": float(np.nanmedian(diff.values) * 1e4)}
print("textbook snapshot cross-check:", xcheck)

# %% [markdown]
# ## Cost model sanity: per-leg cost in bps at the raw open

# %%
cost = per_leg_cost_bps(df).reindex(cal)
cb = (cost.loc[me] * 1e4).stack()
print(cb.describe().round(2))
grp = universe().set_index("symbol").group
(cost.loc[me].median() * 1e4).groupby(grp).median().round(2)

# %% [markdown]
# ## Point-in-time checklist (three levels)

# %%
pit = pd.DataFrame([
    ("Values", "Prices only; feature at t uses closes <= t (truncation test in 03)", "PASS"),
    ("Values", "Non-price data (fundamentals, macro, text)", "N/A (none used in v1)"),
    ("Values", "Adjusted prices are back-adjusted with *future* dividends; returns are invariant but levels are not -> levels never used in dollars", "PASS (dollar quantities use raw prices)"),
    ("Universe", "Eligibility computed from history length and trailing 63d median $ADV at t", "PASS"),
    ("Universe", "Leavers included (closed/merged ETFs)", "FAIL - list chosen in 2025; 0 leavers in sample (survivorship bias)"),
    ("Identifiers", "Ticker changes / reuse handled with effective dates", "PASS (no ticker changes among the 100 symbols in 2006-2023; tickers checked against listing dates)"),
], columns=["Level", "Check", "Result"])
pit.to_csv(TAB / "02_pit_checklist.csv", index=False)
quality.to_csv(TAB / "02_quality.csv")
pd.Series({"meta": meta, "cross_check": xcheck, "eligible_per_month": cov.describe().round(1).to_dict(),
           "flagged_rows": int(flag.sum()), "interior_missing_sessions": int(gaps.sum())}).to_json(TAB / "02_summary.json", indent=2, default_handler=str)
pit
