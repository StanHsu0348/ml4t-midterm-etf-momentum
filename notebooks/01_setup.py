# %% [markdown]
# # 01 · Hypothesis, trading setup v1, evaluation protocol, research ledger
# Written before any feature, model or backtest was computed (Week 1 checkpoint).

# %%
import pandas as pd
import yaml
from etfmom.config import ROOT, TAB, setup
from etfmom.ledger import LEDGER, init_ledger

s = setup()
print(yaml.safe_dump(s["hypothesis"], allow_unicode=True, sort_keys=False))

# %% [markdown]
# ## Evaluation protocol (six design commitments + split plan, calendar dates)

# %%
ev = s["evaluation"]
rows = [
    ("Window", ev["window"] + f" from {ev['first_train_date']}"),
    ("Step size", ev["step"]),
    ("Retraining cadence", ev["retrain_cadence"]),
    ("Label horizon", f"{ev['label_horizon_sessions']} sessions (open t+1 -> open t+22)"),
    ("Label buffer (purge)", f"{ev['label_buffer_sessions']} sessions"),
    ("Feature buffer (embargo)", f"{ev['feature_embargo_sessions']} (training never follows validation)"),
    ("Longest feature lookback", f"{ev['longest_feature_lookback_sessions']} sessions"),
    ("Discovery (feature triage)", " to ".join(ev["discovery_period"])),
    ("HP tuning (fixed split)", f"train <= {ev['tuning_split']['train_end']}, valid {' to '.join(ev['tuning_split']['valid'])}"),
    ("Validation folds", f"{ev['validation_folds'][0]}-{ev['validation_folds'][-1]} (one calendar year each)"),
    ("Holdout (sealed)", " to ".join(ev["holdout"])),
]
protocol = pd.DataFrame(rows, columns=["Commitment", "Value"])
protocol.to_csv(TAB / "protocol.csv", index=False)
protocol

# %%
folds = []
for y in ev["validation_folds"]:
    folds.append({"fold": y, "train_start": ev["first_train_date"], "train_end": f"{y}-01-01 minus {ev['label_buffer_sessions']} sessions",
                  "valid_start": f"{y}-01-01", "valid_end": f"{y}-12-31"})
for y in (2024, 2025):
    folds.append({"fold": f"holdout_{y}", "train_start": ev["first_train_date"], "train_end": f"{y}-01-01 minus {ev['label_buffer_sessions']} sessions",
                  "valid_start": f"{y}-01-01", "valid_end": f"{y}-12-31"})
pd.DataFrame(folds)

# %%
init_ledger()
print(LEDGER.read_text())
