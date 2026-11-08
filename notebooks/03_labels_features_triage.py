# %% [markdown]
# # 03 · Labels, features and the five-gate triage
# Triage runs on the **discovery period 2008-2014 only**, so the validation folds
# (2015-2023) stay untouched by feature selection. Every candidate is logged to
# the research ledger as one trial of the `feature_triage` family.

# %%
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from etfmom.config import ART, FIG, TAB, setup
from etfmom.data import load_panel, per_leg_cost_bps, universe, wide, month_end_sessions
from etfmom.features import REGISTRY, SIGNALS, STATES, build_panel, compute_raw_features
from etfmom.ledger import log_run
from etfmom.metrics import benjamini_hochberg, daily_ic, effective_n, ic_summary, nw_mean

plt.style.use("seaborn-v0_8-whitegrid")
S = setup()
df = load_panel()
panel = build_panel(df)
panel.to_parquet(ART / "panel_dev.parquet")
LABEL = "fwd_open_ret_21d"
d0, d1 = S["evaluation"]["discovery_period"]
disc = panel.loc[d0:d1]
print(panel.shape, "| discovery rows:", len(disc))

# %% [markdown]
# ## Label specification and the close-vs-open convention

# %%
lab = panel[[LABEL, "fwd_close_ret_21d"]].dropna()
gap = (lab["fwd_close_ret_21d"] - lab[LABEL]).abs()
ic_lab = daily_ic(disc, "mom_252_21", "fwd_close_ret_21d")
ic_open = daily_ic(disc, "mom_252_21", LABEL)
dates = panel.index.get_level_values("date").unique()
n_nom = len(lab)
n_eff_overlap = n_nom / S["label"]["horizon_sessions"]
label_tab = pd.DataFrame({
    "item": ["label", "type", "execution anchor", "horizon", "mean |close-close - open-open| per obs (bp)",
             "IC(mom_252_21) close-to-close label", "IC(mom_252_21) next-open label",
             "nominal obs (dev)", "effective obs ~ nominal / horizon", "effective IC days (Bartlett, discovery)"],
    "value": [LABEL, "fixed-horizon continuous return", "fill at open t+1, exit at open t+22", "21 sessions",
              f"{gap.mean()*1e4:.0f}", f"{ic_lab.mean():.4f}", f"{ic_open.mean():.4f}",
              f"{n_nom:,}", f"{n_eff_overlap:,.0f}", f"{effective_n(ic_open):.0f}"]})
label_tab.to_csv(TAB / "03_label_spec.csv", index=False)
label_tab

# %% [markdown]
# ## Gate 1a — point-in-time truncation test
# Recompute every feature on data truncated at date t and require identical
# values at t. A feature that changes when the future is removed is leaking.

# %%
rng = np.random.default_rng(42)
test_dates = sorted(rng.choice(dates[(dates >= "2008-01-01") & (dates <= "2023-06-30")], 6, replace=False))
full_raw = compute_raw_features(df)
mk_full = full_raw.pop("_mkt"); full_raw.pop("_ret21")
trunc_fail = {}
for t in test_dates:
    tr = compute_raw_features(df[df.date <= t])
    mk_t = tr.pop("_mkt"); tr.pop("_ret21")
    for k, v in tr.items():
        a = v.loc[t]; b = full_raw[k].loc[t].reindex(a.index)
        assert full_raw[k].loc[t].drop(a.index).isna().all()
        ok = np.allclose(a.values, b.values, equal_nan=True, rtol=1e-9, atol=1e-12)
        trunc_fail[k] = trunc_fail.get(k, 0) + (not ok)
    for k in mk_t:
        ok = np.isclose(mk_t[k].loc[t], mk_full[k].loc[t], equal_nan=True)
        trunc_fail[k] = trunc_fail.get(k, 0) + (not ok)
trunc = pd.Series(trunc_fail, name="truncation_failures")
print("dates tested:", [str(pd.Timestamp(x).date()) for x in test_dates])
print(trunc[trunc > 0] if (trunc > 0).any() else "all features identical under truncation -> PASS")
# label sanity: the label must use FUTURE opens; the planted-leak check is the opposite test
assert trunc.sum() == 0

# %% [markdown]
# ## Correlation clustering (dedupe) — order of preference is the register order, not IC

# %%
corr = disc[SIGNALS].sample(60000, random_state=0).corr(method="spearman")
dist = 1 - corr.abs()
Z = linkage(squareform(dist.values, checks=False), "average")
cl = pd.Series(fcluster(Z, t=0.2, criterion="distance"), index=SIGNALS, name="cluster")  # |rho| >= 0.8
rep = cl.groupby(cl).apply(lambda s: s.index[0])           # first in register order
dup_of = {f: rep[cl[f]] for f in SIGNALS if rep[cl[f]] != f}
print("clusters:", cl.nunique(), "| duplicates:", dup_of)
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(SIGNALS)), SIGNALS, rotation=90, fontsize=7); ax.set_yticks(range(len(SIGNALS)), SIGNALS, fontsize=7)
ax.set_title("Q: which candidate signals are the same bet? (Spearman, discovery)")
fig.colorbar(im, fraction=0.04); fig.tight_layout(); fig.savefig(FIG / "03_feature_corr.png", dpi=150)

# %% [markdown]
# ## Gates 1b-5 for every candidate signal

# %%
cost = per_leg_cost_bps(df)
me = month_end_sessions(pd.DatetimeIndex(dates))
me_disc = me[(me >= d0) & (me <= d1)]
rt_cost = 2 * float(np.nanmedian(cost.reindex(me_disc).values))       # round-trip, fraction
yr_blocks = sorted(set(disc.index.get_level_values("date").year))

def placebo_ic(feat, k=21):
    """Timing placebo: feature value from k sessions earlier, same label."""
    lagged = disc[feat].groupby(level="symbol").shift(k)
    tmp = disc[[LABEL]].assign(pl=lagged)
    return daily_ic(tmp, "pl", LABEL).mean()

rows = []
for feat in SIGNALS:
    sign = int(REGISTRY.set_index("feature").loc[feat, "expected_sign"])
    ic = daily_ic(disc, feat, LABEL)
    s = ic_summary(ic)
    yearly = ic.groupby(ic.index.year).mean()
    same_sign_share = float((np.sign(yearly) == sign).mean())
    decay = {h: daily_ic(disc, feat, f"fwd_open_ret_{h}d").mean() for h in (5, 21, 63)}
    pl = placebo_ic(feat, 21)
    # shape: quintile mean label on month-end decision dates
    m = disc.loc[disc.index.get_level_values("date").isin(me_disc), [feat, LABEL]].dropna()
    m["q"] = m.groupby(level="date")[feat].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    qm = m.groupby("q")[LABEL].mean()
    mono = stats.spearmanr(qm.index, qm.values).statistic * sign
    spread = (qm.iloc[-1] - qm.iloc[0]) * sign                           # monthly long-short spread in the hypothesised direction
    # feasibility: monthly turnover of the hypothesised top quintile
    top = m["q"].eq(4 if sign > 0 else 0).unstack().fillna(False)
    turn = (top.astype(int).diff().abs().sum(axis=1) / (2 * top.sum(axis=1).clip(lower=1))).iloc[1:].mean()
    cost_month = turn * rt_cost * 2                                        # both legs of the spread
    rows.append({"feature": feat, "expected_sign": sign, "ic_mean": s["mean"], "ic_lo": s["lo"], "ic_hi": s["hi"],
                 "p_nw": s["p"], "icir_daily": s["icir_daily"], "same_sign_years": same_sign_share,
                 "ic_5d": decay[5], "ic_21d": decay[21], "ic_63d": decay[63], "placebo_ic_lag21": pl,
                 "quintile_monotonicity": mono, "q5_q1_spread_month": spread, "turnover_month": turn,
                 "cost_month": cost_month, "duplicate_of": dup_of.get(feat, "")})
tri = pd.DataFrame(rows).set_index("feature")
tri["bh_sig"] = benjamini_hochberg(tri["p_nw"]).reindex(tri.index).fillna(False)
tri["effect_d"] = tri["ic_mean"] / disc.groupby(level="date").size().pipe(lambda n: 1 / np.sqrt(n.mean() - 3))  # IC in Fisher-z SE units

def decide(r):
    g1 = trunc.get(r.name, 0) == 0
    right_dir = np.sign(r.ic_mean) == r.expected_sign
    sig = r.bh_sig and ((r.ic_lo > 0) if r.expected_sign > 0 else (r.ic_hi < 0))
    slow = abs(r.placebo_ic_lag21) >= 0.5 * abs(r.ic_mean) and np.sign(r.placebo_ic_lag21) == np.sign(r.ic_mean)
    g3 = r.quintile_monotonicity >= 0.9
    g4 = r.q5_q1_spread_month > 2 * r.cost_month
    notes = []
    if not g1: return "STOP", "fails truncation test"
    if r.duplicate_of: return "STOP", f"duplicate of {r.duplicate_of} (|rho|>=0.8)"
    if not right_dir and r.same_sign_years < 0.5:
        return "STOP", "sign opposite to mechanism; no stable support"
    if slow: notes.append("slow state proxy (placebo survives)")
    if not g3: notes.append("non-monotone quintiles")
    if not g4: notes.append("spread < 2x cost")
    if not sig: notes.append("CI includes 0 / not BH-significant")
    if sig and g3 and g4 and not slow: return "PROCEED", "all gates pass"
    if right_dir and r.same_sign_years >= 4 / 7: return "REVISE", "; ".join(notes)
    return "STOP", "; ".join(notes) or "weak"

dec = tri.apply(decide, axis=1, result_type="expand"); dec.columns = ["decision", "reason"]
tri = tri.join(dec)
tri.round(4).to_csv(TAB / "03_triage.csv")
for feat, r in tri.iterrows():
    log_run("feature_triage", feat, "discovery", {"label": LABEL, "expected_sign": int(r.expected_sign)},
            f"{d0}..{d1}", {k: float(r[k]) for k in ["ic_mean", "ic_lo", "ic_hi", "p_nw", "same_sign_years", "quintile_monotonicity", "q5_q1_spread_month", "cost_month"]},
            r.decision, r.reason)
log_run("feature_triage", "correlation_dedupe", "discovery", {"threshold_abs_rho": 0.8, "linkage": "average"}, f"{d0}..{d1}",
        {"n_clusters": int(cl.nunique())}, "applied", "")
tri[["ic_mean", "ic_lo", "ic_hi", "same_sign_years", "placebo_ic_lag21", "quintile_monotonicity", "q5_q1_spread_month", "cost_month", "bh_sig", "decision", "reason"]].round(4)

# %% [markdown]
# ## State variables: regime heterogeneity of the primary signal (mechanism test)
# Terciles use **expanding, point-in-time** thresholds (each date compared with
# the state's own history up to that date).

# %%
def pit_tercile(x: pd.Series) -> pd.Series:
    s = x.groupby(level="date").first()
    lo = s.expanding(252).quantile(1 / 3); hi = s.expanding(252).quantile(2 / 3)
    t = pd.Series(np.where(s <= lo, 0, np.where(s <= hi, 1, 2)), index=s.index).where(lo.notna())
    return t

ic_mom = daily_ic(panel.loc[:"2014-12-31"], "mom_252_21", LABEL)
state_rows = []
for st in [c for c in STATES if c != "asset_rvol_pct"]:
    terc = pit_tercile(panel[st]).reindex(ic_mom.index)
    for k in (0, 1, 2):
        x = ic_mom[terc == k]
        s = nw_mean(x)
        state_rows.append({"state": st, "tercile": ["low", "mid", "high"][k], "ic_mean": s["mean"], "ic_lo": s["lo"], "ic_hi": s["hi"], "days": s["T"]})
st_tab = pd.DataFrame(state_rows)
st_tab.round(4).to_csv(TAB / "03_state_heterogeneity.csv", index=False)
for st in st_tab.state.unique():
    log_run("feature_triage", st, "discovery_state_tercile", {"signal": "mom_252_21"}, f"2008..{d1}",
            {r.tercile: float(r.ic_mean) for r in st_tab[st_tab.state == st].itertuples()}, "keep_as_state", "regime heterogeneity")
st_tab.pivot(index="state", columns="tercile", values="ic_mean")[["low", "mid", "high"]].round(4)

# %% [markdown]
# ## Mechanism plausibility for the top-3 signals: DAG + four cheap checks

# %%
cand = tri[tri.decision.isin(["PROCEED", "REVISE"])].copy()
cand["abs_t"] = (cand.ic_mean / ((cand.ic_hi - cand.ic_lo) / 3.92)).abs()
top3 = cand.sort_values("abs_t", ascending=False).index[:3].tolist()
grp = universe().set_index("symbol").group
checks = []
for feat in top3:
    sign = int(REGISTRY.set_index("feature").loc[feat, "expected_sign"])
    base = daily_ic(disc, feat, LABEL).mean()
    pl5 = placebo_ic(feat, 5); pl21 = placebo_ic(feat, 21)
    # shared-driver control: residualise the raw feature on beta_spy_126 + asset-group dummies per date
    tmp = disc[[feat + "__raw", "beta_spy_126__raw", LABEL]].dropna().copy()
    tmp["grp"] = tmp.index.get_level_values("symbol").map(grp)
    def resid(x):
        X = pd.get_dummies(x["grp"], drop_first=False).astype(float)
        X["beta"] = x["beta_spy_126__raw"]
        y = x[feat + "__raw"]
        b, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
        return pd.Series(y.values - X.values @ b, index=x.index)
    tmp["res"] = tmp.groupby(level="date", group_keys=False).apply(resid)
    ic_res = daily_ic(tmp, "res", LABEL).mean()
    terc = pit_tercile(panel["mkt_rvol_pct"]).reindex(daily_ic(disc, feat, LABEL).index)
    icd = daily_ic(disc, feat, LABEL)
    reg = {k: icd[terc == i].mean() for i, k in enumerate(["calm", "mid", "stressed"])}
    checks.append({"feature": feat, "ic": base, "placebo_lag5": pl5, "placebo_lag21": pl21, "ic_after_beta_group_control": ic_res,
                   "ic_calm": reg["calm"], "ic_mid": reg["mid"], "ic_stressed": reg["stressed"], "event_time": "N/A (not event-based)"})
chk = pd.DataFrame(checks).set_index("feature")
chk.round(4).to_csv(TAB / "03_mechanism_checks.csv")
chk.round(4)

# %%
def draw_dag(ax, title, nodes, edges):
    for n, (x, y, kind) in nodes.items():
        col = {"T": "#4C72B0", "Y": "#55A868", "conf": "#C44E52", "med": "#8172B2", "coll": "#CCB974"}[kind]
        ax.text(x, y, n, ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle="round", fc=col, alpha=0.25, ec=col))
    for a, b in edges:
        (x0, y0, _), (x1, y1, _) = nodes[a], nodes[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="0.3", shrinkA=18, shrinkB=18))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_title(title, fontsize=9)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, feat in zip(axes, top3):
    nodes = {feat: (0.2, 0.5, "T"), "fwd return": (0.8, 0.5, "Y"), "market beta /\nasset class (control)": (0.5, 0.88, "conf"),
             "vol regime\n(control)": (0.5, 0.12, "conf"), "fund flows\n(mediator: do NOT control)": (0.5, 0.5, "med"),
             "AUM / survival\n(collider: NEVER control)": (0.85, 0.12, "coll")}
    edges = [(feat, "fund flows\n(mediator: do NOT control)"), ("fund flows\n(mediator: do NOT control)", "fwd return"),
             ("market beta /\nasset class (control)", feat), ("market beta /\nasset class (control)", "fwd return"),
             ("vol regime\n(control)", feat), ("vol regime\n(control)", "fwd return"),
             (feat, "AUM / survival\n(collider: NEVER control)"), ("fwd return", "AUM / survival\n(collider: NEVER control)")]
    draw_dag(ax, f"DAG: {feat}", nodes, edges)
fig.tight_layout(); fig.savefig(FIG / "03_dags.png", dpi=150)

# %%
selected = tri[tri.decision.isin(["PROCEED", "REVISE"])].index.tolist()
pd.Series({"selected_signals": selected, "states": STATES, "top3": top3}).to_json(ART / "selected_features.json")
print("model inputs ->", selected, "+ states", STATES)
REGISTRY.to_csv(TAB / "03_feature_register.csv", index=False)
