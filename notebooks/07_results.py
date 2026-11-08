# %% [markdown]
# # 07 · Results: headline table, six required figures, diagnosis
# Reads only saved artifacts and tables from notebooks 04-06 (never the data).

# %%
import json
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etfmom.config import ART, FIG, TAB, setup
from etfmom.metrics import daily_ic, ic_summary, sharpe
from etfmom.models import LABEL

plt.style.use("seaborn-v0_8-whitegrid")
S = setup()
spec = json.loads((ART / "frozen_spec.json").read_text())
dep = spec["deployed"]; DEP = "deployed_" + dep
val_ic = pd.read_csv(TAB / "04_ic_summary.csv", index_col=0)
ho_ic = pd.read_csv(TAB / "06_holdout_ic.csv", index_col=0)
val_perf = pd.read_csv(TAB / "05_performance_validation.csv", index_col=0)
ho_perf = pd.read_csv(TAB / "06_holdout_performance.csv", index_col=0)
val_inf = json.loads((TAB / "05_inference.json").read_text())
ho_inf = json.loads((TAB / "06_holdout_inference.json").read_text())
val_cost = pd.read_csv(TAB / "05_costs.csv", index_col=0)
val_ov = pd.read_csv(TAB / "05_overlay.csv", index_col=0)
vr = pd.read_parquet(ART / "validation_returns.parquet"); hr = pd.read_parquet(ART / "holdout_returns.parquet")
vg = pd.read_parquet(ART / "validation_gross.parquet"); hg = pd.read_parquet(ART / "holdout_gross.parquet")
ho_ic_d = ho_ic.loc[dep]

# %% [markdown]
# ## Headline table (validation vs holdout vs benchmark)

# %%
def ci(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
from etfmom.metrics import perf_summary
ov_h = perf_summary(hr["deployed + vol target"])
head = pd.DataFrame([
    ("Signal", "Daily IC [NW 95% CI]", ci(val_ic.loc[dep, "IC"], val_ic.loc[dep, "CI_lo"], val_ic.loc[dep, "CI_hi"]), ci(ho_ic_d["mean"], ho_ic_d["lo"], ho_ic_d["hi"]), "-"),
    ("Signal", "ICIR (daily) / share of positive folds", f"{val_ic.loc[dep, 'ICIR(daily)']:.3f} / {val_ic.loc[dep, 'share_pos_folds']:.2f}", f"{ho_ic_d['icir_daily']:.3f} / {ho_ic_d['share_pos_folds']:.2f}", "-"),
    ("Portfolio", "Gross Sharpe", f"{val_perf.loc[DEP, 'gross_sharpe']:.2f}", f"{ho_perf.loc[DEP, 'gross_sharpe']:.2f}",
     f"EW {val_perf.loc['EW_all_eligible', 'net_sharpe']:.2f} | 60/40 {val_perf.loc['60/40 SPY/AGG', 'net_sharpe']:.2f} | SPY {val_perf.loc['SPY', 'net_sharpe']:.2f} (val)"),
    ("Portfolio", "Max drawdown / 2-way turnover p.a.", f"{val_perf.loc[DEP, 'max_dd']:.1%} / {val_perf.loc[DEP, 'turnover_2way_ann']:.1f}x", f"{ho_perf.loc[DEP, 'max_dd']:.1%} / {ho_perf.loc[DEP, 'turnover_2way_ann']:.1f}x",
     f"EW {ho_perf.loc['EW_all_eligible', 'net_sharpe']:.2f} | 60/40 {ho_perf.loc['60/40 SPY/AGG', 'net_sharpe']:.2f} | SPY {ho_perf.loc['SPY', 'net_sharpe']:.2f} (holdout)"),
    ("Costs", "Net Sharpe [95% CI] at assumed cost", ci(val_perf.loc[DEP, "net_sharpe"], val_perf.loc[DEP, "net_sr_lo"], val_perf.loc[DEP, "net_sr_hi"], 2),
     ci(ho_perf.loc[DEP, "net_sharpe"], ho_perf.loc[DEP, "net_sr_lo"], ho_perf.loc[DEP, "net_sr_hi"], 2), "-"),
    ("Costs", "Break-even cost (bp per leg)", f"{val_cost.loc[DEP, 'breakeven_bps_per_leg_vs_cash']:.0f}", f"{ho_inf['breakeven_bps_per_leg']:.0f}", "-"),
    ("Risk overlay", "Net Sharpe / MDD with 10% vol target", f"{val_ov.loc['with vol-target overlay', 'sharpe']:.2f} / {val_ov.loc['with vol-target overlay', 'max_dd']:.1%}",
     f"{ov_h['sharpe']:.2f} / {ov_h['max_dd']:.1%}", "-"),
    ("Inference", "Deflated Sharpe (trials = N) / min. track record", f"{val_inf['dsr_all_trials']:.2f} (N={val_inf['n_trials_all']}) / {val_inf['min_track_record_years']:.0f} y",
     f"{ho_inf['dsr_all_trials']:.2f} (N={ho_inf['n_trials_all']}) / {ho_inf['min_track_record_years']:.0f} y", "-"),
], columns=["Stage", "Metric", "Validation 2015-2023", "Holdout 2024-2025", "Benchmark"])
head.to_csv(TAB / "07_headline.csv", index=False)
head

# %%
v_ok = val_ic.loc[dep, "CI_lo"] > 0
h_ic, h_sr, v_sr = ho_ic_d["mean"], ho_perf.loc[DEP, "net_sharpe"], val_perf.loc[DEP, "net_sharpe"]
if h_ic <= 0 or h_sr <= 0:
    pattern = "decay or sign flip"
elif not v_ok and ho_ic_d["lo"] <= 0:
    pattern = "never resolved"
elif h_sr >= v_sr and ho_ic_d["lo"] > 0:
    pattern = "strengthen"
elif ho_ic_d["lo"] <= 0 or ho_perf.loc[DEP, "net_sr_lo"] <= 0:
    pattern = "hold (direction), but holdout interval spans zero"
else:
    pattern = "hold"
print("Chapter-20 pattern:", pattern, "| confirmatory test passed:", ho_inf["confirmatory_test_passed"])

# %% [markdown]
# ## Figure 1 — equity curves, net of costs, validation/holdout boundary marked

# %%
H0 = pd.Timestamp(S["evaluation"]["holdout"][0])
cols = [DEP, "rule_mom_252_21", "EW_all_eligible", "60/40 SPY/AGG"]
allr = pd.concat([vr[cols + ["deployed + vol target"]], hr[cols + ["deployed + vol target"]]]).sort_index()
allr = allr[~allr.index.duplicated(keep="last")]
fig, ax = plt.subplots(figsize=(9, 3.8))
for c in cols + ["deployed + vol target"]:
    ax.plot((1 + allr[c].fillna(0)).cumprod(), label=c, ls="--" if "vol" in c else "-")
ax.axvline(H0, color="k", lw=1.2); ax.text(H0, ax.get_ylim()[1] * 0.95, "  holdout ->", fontsize=8, va="top")
ax.set_yscale("log"); ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Q: does the deployed model beat its non-ML baseline and passive benchmarks after costs?", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "07_fig1_equity.png", dpi=160)

# %% [markdown]
# ## Figure 2 — rolling 3-month IC with the zero line

# %%
vp = pd.read_parquet(ART / "validation_predictions.parquet"); hpp = pd.read_parquet(ART / "holdout_predictions.parquet")
hpp = hpp[~hpp.is_prior_decision]
ic_all = {k: pd.concat([daily_ic(vp, k, LABEL), daily_ic(hpp, k, LABEL)]).sort_index() for k in (dep, "rule")}
fig, ax = plt.subplots(figsize=(9, 3.2))
for k, v in ic_all.items():
    ax.plot(v.rolling(63).mean(), label=f"{k}: 63-session rolling mean daily IC")
ax.axhline(0, color="k", lw=0.8); ax.axvline(H0, color="k", lw=1.2); ax.legend(fontsize=8)
ax.set_title("Q: is the predictive signal stable through time, including the holdout?", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "07_fig2_rolling_ic.png", dpi=160)

# %% [markdown]
# ## Figure 3 — gross-to-net Sharpe with the break-even cost

# %%
gn = pd.DataFrame({"validation gross": [sharpe(vg[DEP]), sharpe(vg["rule_mom_252_21"])],
                   "validation net": [sharpe(vr[DEP]), sharpe(vr["rule_mom_252_21"])],
                   "holdout gross": [sharpe(hg[DEP]), sharpe(hg["rule_mom_252_21"])],
                   "holdout net": [sharpe(hr[DEP]), sharpe(hr["rule_mom_252_21"])]}, index=[dep + " (deployed)", "rule 12-1"])
ax = gn.plot.bar(figsize=(8, 3.4), rot=0, color=["#9ecae1", "#3182bd", "#fdae6b", "#e6550d"])
ax.axhline(0, color="k", lw=0.8)
ax.set_title(f"Q: how much do costs take? Break-even {val_cost.loc[DEP, 'breakeven_bps_per_leg_vs_cash']:.0f} bp/leg (val), "
             f"{ho_inf['breakeven_bps_per_leg']:.0f} bp/leg (holdout) vs ~{val_cost.loc[DEP, 'assumed_median_bps_per_leg']:.1f} bp assumed", fontsize=8.5)
plt.tight_layout(); plt.savefig(FIG / "07_fig3_gross_net.png", dpi=160)
gn.round(2)

# %% [markdown]
# ## Figure 4 — regime slice (vol x trend, PIT thresholds)

# %%
rs_v = pd.read_csv(TAB / "05_regime_slice.csv"); rs_h = pd.read_csv(TAB / "06_holdout_regime_slice.csv")
rs_v["period"], rs_h["period"] = "validation", "holdout"
rs = pd.concat([rs_v, rs_h])
rs = rs[rs.strategy.isin(["deployed", DEP])]
piv = rs.pivot_table(index="regime", columns="period", values=["sharpe", "days"])
fig, ax = plt.subplots(figsize=(8, 3.2))
p2 = rs.pivot_table(index="regime", columns="period", values="sharpe")[["validation", "holdout"]]
p2.plot.barh(ax=ax)
for i, reg in enumerate(p2.index):
    d = rs[rs.regime == reg].set_index("period")["days"]
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > -12 else -12, i, f"  n={int(d.get('validation', 0))}/{int(d.get('holdout', 0))} days", va="center", fontsize=7)
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("annualised Sharpe of the deployed strategy")
ax.set_title("Q: in which market regime does the strategy make / lose money?", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "07_fig4_regime.png", dpi=160)
piv.round(2)

# %% [markdown]
# ## Figure 5 — SHAP beeswarm (from notebook 04): `results/figures/04_shap_beeswarm.png`
# ## Figure 6 — drawdown with the kill-switch ladder

# %%
r = allr[DEP].fillna(0)
eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.fill_between(dd.index, dd * 100, 0, color="C3", alpha=0.45)
for lvl in S["risk"]["kill_switch_ladder"]:
    ax.axhline(-lvl["dd"] * 100, ls="--", lw=0.8, color="k")
    ax.text(dd.index[-1], -lvl["dd"] * 100 + 0.5, f"{lvl['action']} {lvl['dd']:.0%}", fontsize=7, ha="right")
ax.axvline(H0, color="k", lw=1.2); ax.set_ylabel("drawdown %")
ax.set_title("Q: when would the kill-switch ladder have fired?", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "07_fig6_drawdown.png", dpi=160)
json.dump({"pattern": pattern, "confirmatory_passed": ho_inf["confirmatory_test_passed"]}, open(TAB / "07_pattern.json", "w"))
