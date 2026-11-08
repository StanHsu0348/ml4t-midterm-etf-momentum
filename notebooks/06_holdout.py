# %% [markdown]
# # 06 · The single holdout run (2024-01-01 .. 2025-12-31)
# Runs the frozen specification exactly once. The gate refuses a second touch;
# a grader's reproducibility re-run sets `HOLDOUT_RERUN=1`, which is logged in
# `results/holdout_access.log`. Nothing in this notebook is tuned.

# %%
import json
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from etfmom import backtest as bt
from etfmom.config import ART, TAB, setup
from etfmom.data import eligibility, load_panel_with_holdout, month_end_sessions
from etfmom.features import build_panel
from etfmom.ledger import log_run, trial_count
from etfmom.metrics import daily_ic, dd_stats, dsr, ic_summary, min_trl_bailey, min_trl_years, perf_summary, psr, sharpe_ci
from etfmom.models import LABEL, LGBMModel, RidgeModel, RuleModel, ensemble, make_fold, train_rows

S = setup()
spec = json.loads((ART / "frozen_spec.json").read_text())
import hashlib
from etfmom.config import ROOT
_h = hashlib.sha256()
for fp in spec["files"]:
    _h.update(fp.encode()); _h.update((ROOT / fp).read_bytes())
assert _h.hexdigest()[:16] == spec["spec_hash"], "code changed after the freeze -> refusing to run the holdout"
hp, deployed = spec["hyperparams"], spec["deployed"]
SIGNALS, FEATS = hp["signals"], hp["signals"] + hp["states"]
rerun = os.environ.get("HOLDOUT_RERUN") == "1"
df = load_panel_with_holdout(reason="single confirmatory run of frozen spec" if not rerun else "reproducibility re-run", allow_rerun=rerun)
print("spec:", spec["spec_hash"], "| deployed:", deployed, "| rerun:", rerun, "| last date:", df.date.max().date())

# %%
panel = build_panel(df)
sessions = panel.index.get_level_values("date").unique().sort_values()
me = month_end_sessions(sessions)
H0, H1 = S["evaluation"]["holdout"]
preds = []
for y in (2024, 2025):
    f = make_fold(f"holdout_{y}", sessions, f"{y}-01-01", f"{y}-12-31")
    tr = train_rows(panel, f, sessions)
    assert tr.index.get_level_values("date").max() <= f.train_end < f.valid_start
    prior = me[me < f.valid_start][-1]
    x = pd.concat([panel.loc[[prior]], panel.loc[f.valid_start:f.valid_end]])
    o = x[[LABEL]].copy()
    o["rule"] = RuleModel().predict(x)
    o["ridge"] = RidgeModel(SIGNALS, hp["ridge_alpha"]).fit(tr).predict(x)
    o["lgbm"] = LGBMModel(FEATS, hp["lgbm_params"], hp["lgbm_rounds"]).fit(tr).predict(x)
    o["ensemble"] = ensemble({"rule": o["rule"], "ridge": o["ridge"], "lgbm": o["lgbm"]})
    o["fold"] = f.name
    o["is_prior_decision"] = o.index.get_level_values("date") == prior
    preds.append(o)
    print(f.name, "| train_end", f.train_end.date(), "| train rows", len(tr))
allh = pd.concat(preds)
allh.to_parquet(ART / "holdout_predictions.parquet")
# a month-end that fills in year Y is scored by the fold-Y model -> keep the later fold's row
hp_pred = allh[~allh.index.duplicated(keep="last")]

# %% [markdown]
# ## Signal stage on the holdout

# %%
in_h = allh[~allh.is_prior_decision]   # each fold's own validation-year rows
ics = {k: daily_ic(in_h, k, LABEL) for k in ("rule", "ridge", "lgbm", "ensemble")}
ic_tab = pd.DataFrame({k: ic_summary(v) for k, v in ics.items()}).T[["mean", "lo", "hi", "p", "icir_daily", "share_pos_folds", "T"]]
ic_tab.to_csv(TAB / "06_holdout_ic.csv"); ic_tab.round(4)

# %% [markdown]
# ## Portfolio, cost and overlay stages on the holdout (identical protocol)

# %%
mkt = bt.market_from_panel(df)
scores = {k: hp_pred[k].unstack().reindex(me).dropna(how="all") for k in ("rule", "ridge", "lgbm", "ensemble")}
elig = eligibility(df)
runs = {
    "deployed_" + deployed: bt.run(scores[deployed], mkt, H0, H1),
    "rule_mom_252_21": bt.run(scores["rule"], mkt, H0, H1),
    "ridge": bt.run(scores["ridge"], mkt, H0, H1),
    "ensemble": bt.run(scores["ensemble"], mkt, H0, H1),
    "deployed + vol target": bt.run(scores[deployed], mkt, H0, H1, vol_target=S["risk"]["vol_target_annual"]),
    "EW_all_eligible": bt.run(None, mkt, H0, H1, universe_mask=elig),
    "60/40 SPY/AGG": bt.run(None, mkt, H0, H1, fixed_weights={"SPY": 0.6, "AGG": 0.4}),
    "SPY": bt.run(None, mkt, H0, H1, fixed_weights={"SPY": 1.0}),
}
g0 = bt.run(scores[deployed], mkt, H0, H1, cost_mult=0.0)
rows = []
for k, r in runs.items():
    p, g = perf_summary(r["ret"]), perf_summary(r["gross"])
    rows.append({"strategy": k, "gross_sharpe": g["sharpe"], "net_sharpe": p["sharpe"], "net_sr_lo": p["sr_lo"], "net_sr_hi": p["sr_hi"],
                 "net_cagr": p["cagr"], "vol": p["vol"], "max_dd": p["max_dd"], "turnover_2way_ann": bt.annual_turnover(r)})
perf = pd.DataFrame(rows).set_index("strategy")
perf.to_csv(TAB / "06_holdout_performance.csv")
pd.DataFrame({k: v["ret"] for k, v in runs.items()}).to_parquet(ART / "holdout_returns.parquet")
perf.round(3)

# %%
r = runs["deployed_" + deployed]["ret"]
ci = sharpe_ci(r)
n_all = trial_count()
be = bt.breakeven_cost(g0) * 1e4
inf = {"net_sharpe": ci["sr"], "ci_lo": ci["lo"], "ci_hi": ci["hi"], "psr_vs_0": psr(r), "dsr_all_trials": dsr(r, n_all)["dsr"],
       "n_trials_all": n_all, "min_track_record_years": min_trl_years(ci["sr"]), "min_trl_years_bailey_ldp": min_trl_bailey(r), "breakeven_bps_per_leg": be,
       "ic_mean": float(ics[deployed].mean()), "ic_lo": float(ic_tab.loc[deployed, "lo"]), "ic_hi": float(ic_tab.loc[deployed, "hi"])}
passed = bool(inf["ic_mean"] > 0 and inf["net_sharpe"] > 0)
inf["confirmatory_test_passed"] = passed
pd.Series(inf).to_json(TAB / "06_holdout_inference.json", indent=2)
if not rerun:
    log_run("holdout", "confirmatory_test", "single_touch", {"spec_hash": spec["spec_hash"], "deployed": deployed}, f"{H0}..{H1}",
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in inf.items()},
            "PASS" if passed else "FAIL", spec["confirmatory_test"])
pd.Series(inf)

# %% [markdown]
# ## Holdout risk: regime slice (PIT thresholds), VaR/CVaR, drawdown

# %%
spy_close = mkt.close_adj["SPY"]
spy_vol = np.log(spy_close).diff().rolling(21).std() * np.sqrt(252)
vol_reg = (spy_vol > spy_vol.expanding(252).median()).map({True: "high vol", False: "low vol"}).shift(2)
trend_reg = (spy_close > spy_close.rolling(200).mean()).map({True: "up trend", False: "down trend"}).shift(2)
reg = (vol_reg + " / " + trend_reg)
from etfmom.metrics import sharpe
rows = []
for name in ("deployed_" + deployed, "rule_mom_252_21", "EW_all_eligible"):
    rr = runs[name]["ret"]
    for k, x in rr.groupby(reg.reindex(rr.index)):
        rows.append({"strategy": name, "regime": k, "sharpe": sharpe(x), "ann_ret": x.mean() * 252, "days": len(x)})
pd.DataFrame(rows).to_csv(TAB / "06_holdout_regime_slice.csv", index=False)
q = r.quantile(0.05)
risk = {"VaR95": -q, "CVaR95": -r[r <= q].mean(), **{k: str(v) for k, v in dd_stats(r).items()}}
pd.Series(risk).to_json(TAB / "06_holdout_risk.json", indent=2)
pd.DataFrame({k: v["gross"] for k, v in runs.items()}).to_parquet(ART / "holdout_gross.parquet")
pd.DataFrame(rows)
