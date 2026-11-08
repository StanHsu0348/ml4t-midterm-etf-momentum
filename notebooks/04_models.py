# %% [markdown]
# # 04 · Models and validation: baseline ladder, paired comparisons, SHAP, conformal
# * Hyper-parameters are tuned **once** on the fixed tuning split (train <= 2013,
#   valid 2014) and then frozen.
# * Walk-forward validation on 2015-2023: one model per year, expanding window,
#   22-session purge. Holdout rows do not exist in this notebook.

# %%
import json
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap

from etfmom.config import ART, FIG, SEED, TAB, setup
from etfmom.features import STATES
from etfmom.ledger import log_run
from etfmom.metrics import daily_ic, ic_summary, nw_mean
from etfmom.models import (LABEL, LGBMModel, RidgeModel, RuleModel, ensemble, make_target, train_rows,
                           tuning_fold, valid_rows, validation_folds)

plt.style.use("seaborn-v0_8-whitegrid")
S = setup()
panel = pd.read_parquet(ART / "panel_dev.parquet")
sel = json.loads((ART / "selected_features.json").read_text())
SIGNALS = sel["selected_signals"]
FEATS = SIGNALS + STATES
sessions = panel.index.get_level_values("date").unique().sort_values()
print("signals:", SIGNALS); print("states:", STATES)

# %% [markdown]
# ## Tuning split: Ridge alpha sweep (10 values, 9 orders of magnitude)

# %%
tf = tuning_fold(sessions)
print(tf)
tr, va = train_rows(panel, tf, sessions), valid_rows(panel, tf)
alpha_rows = []
for a in S["models"]["ladder"][1]["alpha_grid"]:
    m = RidgeModel(SIGNALS, a).fit(tr)
    ic = daily_ic(va.assign(pred=m.predict(va)), "pred", LABEL)
    s = ic_summary(ic)
    alpha_rows.append({"alpha": a, "ic": s["mean"], "icir": s["icir_daily"]})
    log_run("ridge_alpha", f"alpha={a:g}", "tuning_2014", {"alpha": a, "features": SIGNALS}, "valid 2014", {"ic": s["mean"], "icir": s["icir_daily"]})
alpha_tab = pd.DataFrame(alpha_rows)
best_alpha = float(alpha_tab.loc[alpha_tab.ic.idxmax(), "alpha"])
alpha_tab.to_csv(TAB / "04_ridge_alpha.csv", index=False)
print("chosen alpha:", best_alpha); alpha_tab.round(4)

# %% [markdown]
# ## Tuning split: LightGBM, Optuna TPE, 50 trials, early stopping on 2014

# %%
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    p = {"num_leaves": trial.suggest_int("num_leaves", 4, 31),
         "max_depth": trial.suggest_int("max_depth", 2, 5),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
         "min_child_samples": trial.suggest_int("min_child_samples", 100, 2000, log=True),
         "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0), "bagging_freq": 1,
         "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True)}
    m = LGBMModel(FEATS, p, 2000).fit(tr, va, early_stopping=100)
    ic = daily_ic(va.assign(pred=m.predict(va)), "pred", LABEL).mean()
    trial.set_user_attr("best_iteration", int(m.best_iteration))
    log_run("lgbm_optuna", f"trial_{trial.number}", "tuning_2014", {**p, "best_iteration": int(m.best_iteration)}, "valid 2014", {"ic": float(ic)})
    return ic

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=S["models"]["ladder"][2]["max_optuna_trials"])
best = study.best_trial
LGB_PARAMS = {**best.params, "bagging_freq": 1}
LGB_ROUNDS = int(best.user_attrs["best_iteration"])
print("best IC 2014:", round(best.value, 4), "| rounds:", LGB_ROUNDS, "|", LGB_PARAMS)
trials = study.trials_dataframe()[["number", "value", "user_attrs_best_iteration"]]
trials.to_csv(TAB / "04_optuna_trials.csv", index=False)
json.dump({"ridge_alpha": best_alpha, "lgbm_params": LGB_PARAMS, "lgbm_rounds": LGB_ROUNDS, "signals": SIGNALS, "states": STATES},
          open(ART / "frozen_hyperparams.json", "w"), indent=2)

# %% [markdown]
# ## Walk-forward validation 2015-2023 (frozen hyper-parameters)

# %%
folds = validation_folds(sessions)
fold_tab = pd.DataFrame([vars(f) for f in folds])
fold_tab.to_csv(TAB / "04_fold_dates.csv", index=False)
preds, shap_store, conformal_rows, ridge_coefs = [], {}, [], {}
for f in folds:
    tr, va = train_rows(panel, f, sessions), valid_rows(panel, f)
    assert tr.index.get_level_values("date").max() <= f.train_end < f.valid_start
    out = va[[LABEL]].copy()
    rule = RuleModel().fit(tr); out["rule"] = rule.predict(va)
    rdg = RidgeModel(SIGNALS, best_alpha).fit(tr); out["ridge"] = rdg.predict(va); ridge_coefs[f.name] = rdg.coef_
    gbm = LGBMModel(FEATS, LGB_PARAMS, LGB_ROUNDS).fit(tr); out["lgbm"] = gbm.predict(va)
    out["ensemble"] = ensemble({"rule": out["rule"], "ridge": out["ridge"], "lgbm": out["lgbm"]})
    out["fold"] = f.name
    preds.append(out)
    # SHAP on a fixed-size sample of validation rows
    samp = va[FEATS].sample(min(3000, len(va)), random_state=SEED)
    sv = shap.TreeExplainer(gbm.m).shap_values(samp)
    shap_store[f.name] = (samp, sv)
    # split conformal: fit on core, calibrate on the last 252 training sessions (purged from core)
    tr_dates = tr.index.get_level_values("date")
    cal_start = sessions[sessions.get_loc(f.train_end) - 252]
    core_end = sessions[sessions.get_loc(cal_start) - S["evaluation"]["label_buffer_sessions"] - 1]
    core, cal = tr[tr_dates <= core_end], tr[tr_dates >= cal_start]
    for name, mk in (("ridge", lambda: RidgeModel(SIGNALS, best_alpha)), ("lgbm", lambda: LGBMModel(FEATS, LGB_PARAMS, LGB_ROUNDS))):
        m = mk().fit(core)
        yc, _ = make_target(cal, m.clip); res = (yc - m.predict(cal)).abs()
        yv, _ = make_target(va.dropna(subset=[LABEL]), m.clip); pv = m.predict(va.dropna(subset=[LABEL]))
        for lvl in (0.8, 0.9):
            q = res.quantile(lvl)
            conformal_rows.append({"fold": f.name, "model": name, "level": lvl, "coverage": float(((yv - pv).abs() <= q).mean()), "half_width": float(q)})
    print(f.name, "train rows", len(tr), "| valid rows", len(va))
pred = pd.concat(preds)
pred.to_parquet(ART / "validation_predictions.parquet")
for k in ("rule", "ridge", "lgbm", "ensemble"):
    log_run("model_ladder", k, "walk_forward_2015_2023", {"alpha": best_alpha} if k == "ridge" else ({"params": LGB_PARAMS, "rounds": LGB_ROUNDS} if k == "lgbm" else {}),
            "2015-2023", {"ic": float(daily_ic(pred, k, LABEL).mean())})

# %% [markdown]
# ## Yardstick: mean daily Spearman IC with Newey-West 95% CI (lag 21)

# %%
ics = {k: daily_ic(pred, k, LABEL) for k in ("rule", "ridge", "lgbm", "ensemble")}
summ = pd.DataFrame({k: ic_summary(v) for k, v in ics.items()}).T[["mean", "lo", "hi", "p", "icir_daily", "share_pos_folds", "n_eff", "T"]]
summ.columns = ["IC", "CI_lo", "CI_hi", "p", "ICIR(daily)", "share_pos_folds", "n_eff_days", "days"]
summ.to_csv(TAB / "04_ic_summary.csv")
summ.round(4)

# %%
per_fold = pd.DataFrame({k: v.groupby(v.index.year).mean() for k, v in ics.items()})
per_fold_ci = {k: pd.DataFrame({y: nw_mean(v[v.index.year == y]) for y in per_fold.index}).T for k, v in ics.items()}
fig, ax = plt.subplots(figsize=(9, 3.4))
for i, k in enumerate(per_fold.columns):
    c = per_fold_ci[k]
    ax.errorbar(per_fold.index + (i - 1.5) * 0.15, c["mean"], yerr=[c["mean"] - c["lo"], c["hi"] - c["mean"]], fmt="o", capsize=2, label=k)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("Q: is the signal stable fold by fold? (mean daily IC, NW 95% band)"); ax.set_ylabel("Spearman IC"); ax.legend(ncol=4, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "04_ic_per_fold.png", dpi=150)
per_fold.round(4).to_csv(TAB / "04_ic_per_fold.csv"); per_fold.round(4)

# %% [markdown]
# ## Paired comparisons up the ladder (daily IC difference, NW 95% CI)

# %%
def paired(a, b):
    d = (ics[a] - ics[b]).dropna(); s = nw_mean(d)
    return {"comparison": f"{a} - {b}", "dIC": s["mean"], "lo": s["lo"], "hi": s["hi"], "p": s["p"]}
pairs = pd.DataFrame([paired("ridge", "rule"), paired("lgbm", "ridge"), paired("lgbm", "rule"), paired("ensemble", "rule"), paired("ensemble", "lgbm")])
pairs.round(4).to_csv(TAB / "04_paired.csv", index=False)
# pre-registered deploy rule: climb only if the paired CI lower bound > 0 vs the best simpler rung
deployed = "rule"
if pairs.set_index("comparison").loc["ridge - rule", "lo"] > 0: deployed = "ridge"
best_simpler = deployed
if paired("lgbm", best_simpler)["lo"] > 0: deployed = "lgbm"
if paired("ensemble", deployed)["lo"] > 0 if deployed != "ensemble" else False: deployed = "ensemble"
print("deployed rung:", deployed)
json.dump({"deployed": deployed}, open(ART / "deployed.json", "w"))
log_run("model_ladder", "deploy_decision", "walk_forward_2015_2023", {"rule": S["models"]["deploy_rule"]}, "2015-2023",
        {r.comparison: float(r.dIC) for r in pairs.itertuples()}, f"deploy={deployed}", "paired NW CI lower bound > 0")
pairs.round(4)

# %% [markdown]
# ## SHAP (LightGBM): magnitude, sign consistency, stability across folds

# %%
imp = pd.DataFrame({k: np.abs(sv).mean(axis=0) for k, (X, sv) in shap_store.items()}, index=FEATS)
sign = pd.DataFrame({k: [np.corrcoef(X[c].fillna(X[c].median()), sv[:, j])[0, 1] for j, c in enumerate(FEATS)] for k, (X, sv) in shap_store.items()}, index=FEATS)
stab = imp.rank().corr(method="spearman").values[np.triu_indices(len(imp.columns), 1)].mean()
shap_tab = pd.DataFrame({"mean_abs_shap": imp.mean(axis=1), "rank_mean": imp.rank(ascending=False).mean(axis=1),
                         "sign_corr_mean": sign.mean(axis=1), "sign_consistent_share": (np.sign(sign).eq(np.sign(sign.mean(axis=1)), axis=0)).mean(axis=1)}).sort_values("mean_abs_shap", ascending=False)
shap_tab.round(5).to_csv(TAB / "04_shap.csv")
print(f"mean pairwise Spearman of importance ranks across folds: {stab:.3f}")
X_all = pd.concat([X for X, _ in shap_store.values()]); sv_all = np.vstack([sv for _, sv in shap_store.values()])
plt.figure()
shap.summary_plot(sv_all, X_all, show=False, max_display=14)
plt.title("Q: which inputs drive LightGBM, and in which direction?", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "04_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
shap_tab.round(4)

# %% [markdown]
# ## Conformal prediction intervals: empirical coverage per fold

# %%
conf = pd.DataFrame(conformal_rows)
conf["flag_<80%"] = conf.coverage < 0.8
conf.round(3).to_csv(TAB / "04_conformal.csv", index=False)
conf.pivot_table(index="fold", columns=["model", "level"], values="coverage").round(3)

# %%
pd.DataFrame(ridge_coefs).round(5).to_csv(TAB / "04_ridge_coefs.csv")
pd.DataFrame(ridge_coefs).mean(axis=1).sort_values().round(5)
