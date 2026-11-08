# %% [markdown]
# # 05 · Backtest, baselines, costs, portfolio construction, risk (validation 2015-2023)
# The protocol (engine semantics, costs class, sizing) was fixed in `setup.yaml`
# and `src/etfmom/backtest.py` before this notebook was run. Decisions inside
# fold year Y are scored by the fold-Y model; the December-of-prior-year decision
# that fills on the first session of Y is scored by the fold-Y model too.

# %%
import json
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from etfmom import backtest as bt
from etfmom.config import ART, FIG, TAB, setup
from etfmom.data import eligibility, load_panel, month_end_sessions
from etfmom.features import STATES
from etfmom.ledger import log_run, trial_count
from etfmom.metrics import daily_ic, dd_stats, dsr, ic_summary, min_trl_years, nw_mean, perf_summary, psr, sharpe, sharpe_ci
from etfmom.models import LABEL, LGBMModel, RidgeModel, RuleModel, ensemble, train_rows, validation_folds

plt.style.use("seaborn-v0_8-whitegrid")
S = setup()
df = load_panel()
panel = pd.read_parquet(ART / "panel_dev.parquet")
hp = json.loads((ART / "frozen_hyperparams.json").read_text())
deployed = json.loads((ART / "deployed.json").read_text())["deployed"]
SIGNALS, FEATS = hp["signals"], hp["signals"] + hp["states"]
sessions = panel.index.get_level_values("date").unique().sort_values()
mkt = bt.market_from_panel(df)
pred = pd.read_parquet(ART / "validation_predictions.parquet")
print("deployed rung:", deployed)

# %% [markdown]
# ## Scores on decision dates (month-ends), incl. the prior month-end of each fold

# %%
me = month_end_sessions(mkt.sessions)
extra = []
for f in validation_folds(sessions):
    prior = me[me < f.valid_start][-1]
    tr = train_rows(panel, f, sessions)
    x = panel.loc[[prior]]
    o = x[[LABEL]].copy()
    o["rule"] = RuleModel().predict(x)
    o["ridge"] = RidgeModel(SIGNALS, hp["ridge_alpha"]).fit(tr).predict(x)
    o["lgbm"] = LGBMModel(FEATS, hp["lgbm_params"], hp["lgbm_rounds"]).fit(tr).predict(x)
    o["ensemble"] = ensemble({"rule": o["rule"], "ridge": o["ridge"], "lgbm": o["lgbm"]})
    o["fold"] = f.name
    extra.append(o)
ex = pd.concat(extra)
ex_dates = ex.index.get_level_values("date").unique()
# a month-end that fills in year Y is scored by the fold-Y model (replaces the fold Y-1 score)
allp = pd.concat([pred[~pred.index.get_level_values("date").isin(ex_dates)], ex]).sort_index()
assert not allp.index.duplicated().any()
ic_pred_only = pred  # IC statistics below use the fold-year predictions (identical to notebook 04)
dec_scores = {k: allp[k].unstack().reindex(me).dropna(how="all") for k in ("rule", "ridge", "lgbm", "ensemble")}
V0, V1 = "2015-01-01", "2023-12-29"

# %% [markdown]
# ## Strategies, non-ML baseline and passive benchmarks under the identical protocol

# %%
elig = eligibility(df)
runs = {
    "deployed_" + deployed: bt.run(dec_scores[deployed], mkt, V0, V1),
    "rule_mom_252_21": bt.run(dec_scores["rule"], mkt, V0, V1),
    "ridge": bt.run(dec_scores["ridge"], mkt, V0, V1),
    "ensemble": bt.run(dec_scores["ensemble"], mkt, V0, V1),
    "EW_all_eligible": bt.run(None, mkt, V0, V1, universe_mask=elig),
    "60/40 SPY/AGG": bt.run(None, mkt, V0, V1, fixed_weights={"SPY": 0.6, "AGG": 0.4}),
    "SPY": bt.run(None, mkt, V0, V1, fixed_weights={"SPY": 1.0}),
}
gross0 = {k: bt.run(dec_scores[m], mkt, V0, V1, cost_mult=0.0) for k, m in (("deployed_" + deployed, deployed), ("rule_mom_252_21", "rule"))}
rows = []
for k, r in runs.items():
    p = perf_summary(r["ret"]); g = perf_summary(r["gross"])
    rows.append({"strategy": k, "gross_sharpe": g["sharpe"], "net_sharpe": p["sharpe"], "net_sr_lo": p["sr_lo"], "net_sr_hi": p["sr_hi"],
                 "net_cagr": p["cagr"], "vol": p["vol"], "max_dd": p["max_dd"], "turnover_2way_ann": bt.annual_turnover(r),
                 "avg_hold_months": 12 * 2 / bt.annual_turnover(r) if bt.annual_turnover(r) > 0 else np.nan})
    if k.startswith(("deployed", "rule", "ridge", "ensemble")):
        log_run("strategy_backtest", k, "validation_EW_top10", {"top_n": 10, "allocator": "equal_weight", "cost_mult": 1.0}, f"{V0}..{V1}",
                {"net_sharpe": p["sharpe"], "gross_sharpe": g["sharpe"], "max_dd": p["max_dd"]})
perf = pd.DataFrame(rows).set_index("strategy")
perf.round(3).to_csv(TAB / "05_performance_validation.csv")
perf.round(3)

# %% [markdown]
# ## Cost sensitivity: sweep 0-8x the assumed per-leg cost, and the break-even

# %%
dep = "deployed_" + deployed
sweep = []
for m in S["costs"]["sweep_multipliers"]:
    for name, sc in ((dep, dec_scores[deployed]), ("rule_mom_252_21", dec_scores["rule"])):
        r = bt.run(sc, mkt, V0, V1, cost_mult=m)
        p = perf_summary(r["ret"])
        sweep.append({"strategy": name, "cost_mult": m, "net_sharpe": p["sharpe"], "net_cagr": p["cagr"]})
sweep = pd.DataFrame(sweep)
assumed_bps = float(np.nanmedian(mkt.cost.loc[V0:V1].values) * 1e4)
bps_grid = [0, 5, 10, 25, 50, 75, 100, 150, 200, 300]
flat = []
for b in bps_grid:
    r = bt.run(dec_scores[deployed], mkt, V0, V1, flat_cost=b / 1e4)
    p = perf_summary(r["ret"]); flat.append({"bps_per_leg": b, "net_sharpe": p["sharpe"], "net_cagr": p["cagr"]})
flat = pd.DataFrame(flat)
be = {k: bt.breakeven_cost(v) * 1e4 for k, v in gross0.items()}
ew = runs["EW_all_eligible"]
be_excess = {k: (v["gross"].mean() - ew["ret"].reindex(v["gross"].index).mean()) * 252 / bt.annual_turnover(v) * 1e4 for k, v in gross0.items()}
cost_tab = pd.DataFrame({"breakeven_bps_per_leg_vs_cash": be, "breakeven_bps_per_leg_vs_EW": be_excess,
                         "assumed_median_bps_per_leg": assumed_bps, "turnover_2way_ann": {k: bt.annual_turnover(v) for k, v in gross0.items()}})
cost_tab["regime"] = np.where(cost_tab.breakeven_bps_per_leg_vs_cash < 2 * assumed_bps, "hard floor",
                              np.where(cost_tab.breakeven_bps_per_leg_vs_cash < 5 * assumed_bps, "gray zone", "comfortable"))
cost_tab.round(2).to_csv(TAB / "05_costs.csv"); sweep.round(3).to_csv(TAB / "05_cost_sweep.csv", index=False); flat.round(3).to_csv(TAB / "05_cost_flat.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].plot(flat.bps_per_leg, flat.net_sharpe, "o-"); ax[0].axvline(assumed_bps, color="k", ls=":", label=f"assumed {assumed_bps:.1f} bp")
ax[0].axhline(0, color="0.5", lw=0.8); ax[0].set_xlabel("per-leg cost (bp)"); ax[0].set_ylabel("net Sharpe"); ax[0].legend()
ax[1].plot(flat.bps_per_leg, flat.net_cagr * 100, "o-", color="C1"); ax[1].axhline(0, color="0.5", lw=0.8)
ax[1].axvline(be[dep], color="C3", ls="--", label=f"break-even {be[dep]:.0f} bp"); ax[1].set_xlabel("per-leg cost (bp)"); ax[1].set_ylabel("net CAGR %"); ax[1].legend()
fig.suptitle("Q: how much cost can the deployed strategy absorb? (validation 2015-2023)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "05_cost_sweep.png", dpi=150)
cost_tab.round(2)

# %% [markdown]
# ## Portfolio construction: equal weight vs inverse-vol vs HRP (same signal, same 63d window)
# With top-10 and a 10% cap, every fully invested allocator collapses to equal
# weight, so the controlled comparison uses the top-20 candidate set (cap 10%).

# %%
alloc_rows, alloc_runs = [], {}
for al in ("equal_weight", "inverse_vol", "hrp"):
    r = bt.run(dec_scores[deployed], mkt, V0, V1, top_n=20, allocator=al)
    alloc_runs[al] = r
    p = perf_summary(r["ret"])
    W = pd.DataFrame(r["holdings"]).T.fillna(0)
    hhi = (W ** 2).sum(axis=1)
    alloc_rows.append({"allocator": al + " (top-20)", "net_sharpe": p["sharpe"], "max_dd": p["max_dd"], "turnover_2way_ann": bt.annual_turnover(r),
                       "HHI": hhi.mean(), "effective_bets": (1 / hhi).mean()})
    log_run("allocator", al, "validation_top20", {"top_n": 20, "cov_window": 63, "max_weight": 0.10}, f"{V0}..{V1}", {"net_sharpe": p["sharpe"], "max_dd": p["max_dd"]})
alloc = pd.DataFrame(alloc_rows).set_index("allocator")
alloc.loc["best-worst spread", "net_sharpe"] = alloc.net_sharpe.max() - alloc.net_sharpe.min()
alloc.round(3).to_csv(TAB / "05_allocators.csv"); alloc.round(3)

# %% [markdown]
# ## Risk overlay: 10% vol target from t-1 information — with and without

# %%
ov = bt.run(dec_scores[deployed], mkt, V0, V1, vol_target=S["risk"]["vol_target_annual"])
log_run("risk_overlay", "vol_target_10", "validation", {"target": 0.10, "window": 63}, f"{V0}..{V1}", {"net_sharpe": sharpe(ov["ret"])})
ov_tab = pd.DataFrame({"without overlay": perf_summary(runs[dep]["ret"]), "with vol-target overlay": perf_summary(ov["ret"])}).T
ov_tab["avg_gross_exposure"] = [runs[dep]["exposure"].mean(), ov["exposure"].mean()]
ov_tab.round(3).to_csv(TAB / "05_overlay.csv"); ov_tab.round(3)

# %% [markdown]
# ## Tail and path risk: VaR / CVaR (regime-conditional), drawdown depth, duration, recovery

# %%
r = runs[dep]["ret"]
spy_close = mkt.close_adj["SPY"]
spy_vol = np.log(spy_close).diff().rolling(21).std() * np.sqrt(252)
vol_reg = (spy_vol > spy_vol.expanding(252).median()).map({True: "high vol", False: "low vol"}).shift(1)   # known at t-1
trend_reg = (spy_close > spy_close.rolling(200).mean()).map({True: "up trend", False: "down trend"}).shift(1)
def var_cvar(x):
    q = x.quantile(0.05); return {"VaR95": -q, "CVaR95": -x[x <= q].mean(), "days": len(x)}
risk_rows = {"all": var_cvar(r)}
for k in ("high vol", "low vol"):
    risk_rows[k] = var_cvar(r[vol_reg.reindex(r.index) == k])
var_tab = pd.DataFrame(risk_rows).T
ddx = dd_stats(r)
var_tab.round(4).to_csv(TAB / "05_var.csv")
print("drawdown:", {k: (str(v.date()) if hasattr(v, "date") else v) for k, v in ddx.items()})
var_tab.round(4)

# %%
# rolling 252d beta to SPY with HAC band
spy_r = mkt.oo_ret["SPY"].shift(1).reindex(r.index)
betas = []
for end in r.index[252::21]:
    y = r.loc[:end].iloc[-252:]; x = sm.add_constant(spy_r.loc[y.index])
    m = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    betas.append({"date": end, "beta": m.params.iloc[1], "lo": m.conf_int().iloc[1, 0], "hi": m.conf_int().iloc[1, 1]})
betas = pd.DataFrame(betas).set_index("date")
betas.round(3).to_csv(TAB / "05_rolling_beta.csv")
betas.describe().round(3)

# %% [markdown]
# ## Regime slice (PIT thresholds known at t-1): Sharpe and drawdown by vol x trend

# %%
reg = (vol_reg + " / " + trend_reg).reindex(r.index)
slice_rows = []
for name, rr in (("deployed", r), ("rule", runs["rule_mom_252_21"]["ret"]), ("EW", runs["EW_all_eligible"]["ret"])):
    for k, x in rr.groupby(reg.reindex(rr.index)):
        slice_rows.append({"strategy": name, "regime": k, "sharpe": sharpe(x), "ann_ret": x.mean() * 252, "worst_dd_in_regime": dd_stats(x)["max_dd"], "days": len(x)})
regime_tab = pd.DataFrame(slice_rows)
regime_tab.round(3).to_csv(TAB / "05_regime_slice.csv", index=False)
regime_tab.pivot(index="regime", columns="strategy", values=["sharpe", "days"]).round(2)

# %% [markdown]
# ## Mechanism test on validation: momentum IC by market-vol tercile (pre-registered falsification #4)

# %%
ic_rule = daily_ic(pred.loc[V0:V1], "rule", LABEL)
s_ = panel["mkt_rvol_pct"].groupby(level="date").first()
lo = s_.expanding(252).quantile(1 / 3); hi = s_.expanding(252).quantile(2 / 3)
terc = pd.Series(np.where(s_ <= lo, "calm", np.where(s_ <= hi, "mid", "stressed")), index=s_.index).reindex(ic_rule.index)
mech = pd.DataFrame({k: nw_mean(ic_rule[terc == k]) for k in ("calm", "mid", "stressed")}).T[["mean", "lo", "hi", "T"]]
mech.round(4).to_csv(TAB / "05_mechanism_vol_terciles.csv"); mech.round(4)

# %% [markdown]
# ## Stress tests: COVID 2020 and the 2022 rate shock with 3x spreads

# %%
stress_runs = {"deployed (3x costs)": bt.run(dec_scores[deployed], mkt, V0, V1, cost_mult=S["costs"]["stressed_multiplier"]),
               "deployed + overlay (3x)": bt.run(dec_scores[deployed], mkt, V0, V1, cost_mult=S["costs"]["stressed_multiplier"], vol_target=0.10),
               "rule (3x costs)": bt.run(dec_scores["rule"], mkt, V0, V1, cost_mult=S["costs"]["stressed_multiplier"]),
               "60/40": runs["60/40 SPY/AGG"], "SPY": runs["SPY"]}
st_rows = []
for ep, (a, b) in S["risk"]["stress_episodes"].items():
    for k, v in stress_runs.items():
        x = v["ret"].loc[a:b]
        st_rows.append({"episode": ep, "strategy": k, "return": (1 + x).prod() - 1, "max_dd": dd_stats(x)["max_dd"]})
stress = pd.DataFrame(st_rows)
stress.round(4).to_csv(TAB / "05_stress.csv", index=False)
stress.pivot(index="strategy", columns="episode", values="return").round(4)

# %% [markdown]
# ## Statistical inference on the validation Sharpe

# %%
n_all = trial_count()
n_strat = trial_count(["strategy_backtest", "allocator", "risk_overlay", "model_ladder"])
ci = sharpe_ci(r)
inf = {"net_sharpe": ci["sr"], "ci_lo": ci["lo"], "ci_hi": ci["hi"], "psr_vs_0": psr(r),
       "dsr_all_trials": dsr(r, n_all)["dsr"], "n_trials_all": n_all, "sr0_all": dsr(r, n_all)["sr0_ann"],
       "dsr_strategy_trials": dsr(r, n_strat)["dsr"], "n_trials_strategy": n_strat,
       "skew": dsr(r, n_all)["skew"], "kurtosis": dsr(r, n_all)["kurt"], "min_track_record_years": min_trl_years(ci["sr"])}
pd.Series(inf).to_json(TAB / "05_inference.json", indent=2)
pd.Series(inf).round(3)

# %% [markdown]
# ## Figures: equity curves, drawdown with kill-switch ladder, rolling IC

# %%
fig, ax = plt.subplots(figsize=(9, 3.6))
for k in (dep, "rule_mom_252_21", "EW_all_eligible", "60/40 SPY/AGG"):
    ax.plot((1 + runs[k]["ret"]).cumprod(), label=k)
ax.plot((1 + ov["ret"]).cumprod(), label=dep + " + vol target", ls="--")
ax.set_yscale("log"); ax.legend(fontsize=8); ax.set_title("Q: does the deployed strategy beat its baseline and benchmarks net of costs? (validation)")
fig.tight_layout(); fig.savefig(FIG / "05_equity_validation.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 3))
dd = (1 + r).cumprod(); dd = dd / dd.cummax() - 1
ax.fill_between(dd.index, dd * 100, 0, color="C3", alpha=0.4)
for lvl in S["risk"]["kill_switch_ladder"]:
    ax.axhline(-lvl["dd"] * 100, ls="--", lw=0.8, color="k"); ax.text(dd.index[0], -lvl["dd"] * 100 + 0.4, f"{lvl['action']} ({lvl['dd']:.0%})", fontsize=7)
ax.set_ylabel("drawdown %"); ax.set_title("Q: when would the kill-switch ladder have fired? (validation)")
fig.tight_layout(); fig.savefig(FIG / "05_drawdown_killswitch.png", dpi=150)
def episodes(dd, th):
    """Drawdown episodes (peak -> new peak) whose depth reached th."""
    ep = (dd == 0).cumsum()
    return int((dd.groupby(ep).min() <= -th).sum())
triggers = {lvl["action"]: episodes(dd, lvl["dd"]) for lvl in S["risk"]["kill_switch_ladder"]}
print("kill-switch crossings:", triggers)

ics = {k: daily_ic(pred.loc[V0:V1], k, LABEL) for k in (deployed, "rule")}
fig, ax = plt.subplots(figsize=(9, 3))
for k, v in ics.items():
    ax.plot(v.rolling(63).mean(), label=f"{k} (63d rolling mean IC)")
ax.axhline(0, color="k", lw=0.8); ax.legend(fontsize=8); ax.set_title("Q: is the signal stable through time, not only on average?")
fig.tight_layout(); fig.savefig(FIG / "05_rolling_ic.png", dpi=150)

gn = pd.DataFrame({"gross": {k: perf.loc[k, "gross_sharpe"] for k in (dep, "rule_mom_252_21")}, "net": {k: perf.loc[k, "net_sharpe"] for k in (dep, "rule_mom_252_21")}})
ax = gn.plot.bar(figsize=(6, 3.2), rot=0); ax.set_title(f"Q: how much Sharpe do costs take? break-even {be[dep]:.0f} bp/leg (deployed)", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "05_gross_net.png", dpi=150)
json.dump({"triggers": triggers, "drawdown": {k: str(v) for k, v in ddx.items()}, "breakeven_bps": be, "assumed_bps": assumed_bps},
          open(TAB / "05_misc.json", "w"), indent=2, default=str)
