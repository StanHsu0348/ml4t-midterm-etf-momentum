"""Walk-forward splits with purging, and the model ladder (rule, Ridge, LightGBM, ensemble)."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import SEED, setup

LABEL = "fwd_open_ret_21d"


# ----------------------------------------------------------------------------- splits
@dataclass
class Fold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp        # last training decision date (purged)
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def make_fold(name: str, sessions: pd.DatetimeIndex, valid_start: str, valid_end: str) -> Fold:
    ev = setup()["evaluation"]
    purge = ev["label_buffer_sessions"]
    vs = sessions[sessions >= pd.Timestamp(valid_start)][0]
    i = sessions.get_loc(vs)
    train_end = sessions[i - purge - 1]          # label of train_end exits at open[i-1] < valid_start
    ve_cands = sessions[sessions <= pd.Timestamp(valid_end)]
    return Fold(name, pd.Timestamp(ev["first_train_date"]), train_end, vs, ve_cands[-1])


def validation_folds(sessions: pd.DatetimeIndex) -> list[Fold]:
    return [make_fold(str(y), sessions, f"{y}-01-01", f"{y}-12-31") for y in setup()["evaluation"]["validation_folds"]]


def tuning_fold(sessions: pd.DatetimeIndex) -> Fold:
    v0, v1 = setup()["evaluation"]["tuning_split"]["valid"]
    return make_fold("tuning_2014", sessions, v0, v1)


def train_rows(panel: pd.DataFrame, fold: Fold, sessions: pd.DatetimeIndex, step: int = 5) -> pd.DataFrame:
    """Training sample: every `step`-th session inside the purged window, labelled rows only."""
    tr_dates = sessions[(sessions >= fold.train_start) & (sessions <= fold.train_end)][::-1][::step][::-1]
    d = panel.loc[panel.index.get_level_values("date").isin(tr_dates)]
    return d[d[LABEL].notna()]


def valid_rows(panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    return panel.loc[fold.valid_start:fold.valid_end]


# ----------------------------------------------------------------------------- target
def make_target(d: pd.DataFrame, clip: tuple[float, float] | None = None) -> tuple[pd.Series, tuple[float, float]]:
    y = d[LABEL] - d.groupby(level="date")[LABEL].transform("mean")
    if clip is None:
        clip = (float(y.quantile(0.01)), float(y.quantile(0.99)))   # fitted on the training fold only
    return y.clip(*clip), clip


# ----------------------------------------------------------------------------- models
class RuleModel:
    """Rung 1: rank by 12-1 momentum. No fitting."""
    name = "rule"

    def __init__(self, feature: str = "mom_252_21"):
        self.feature = feature

    def fit(self, d: pd.DataFrame):
        return self

    def predict(self, d: pd.DataFrame) -> pd.Series:
        return d[self.feature]


class RidgeModel:
    """Rung 2: Ridge on XS-ranked signals + signal x market-vol-state interactions."""
    name = "ridge"

    def __init__(self, signals: list[str], alpha: float, state: str = "mkt_rvol_pct"):
        self.signals, self.alpha, self.state = signals, alpha, state

    def _X(self, d: pd.DataFrame) -> pd.DataFrame:
        X = d[self.signals].fillna(0.0)      # rank 0 = cross-sectional median for a missing value
        s = d[self.state].fillna(0.5) - 0.5
        inter = X.mul(s, axis=0).add_suffix("_x_vol")
        return pd.concat([X, inter], axis=1)

    def fit(self, d: pd.DataFrame):
        y, self.clip = make_target(d)
        X = self._X(d)
        self.scaler = StandardScaler().fit(X)                    # training fold only
        self.m = Ridge(alpha=self.alpha).fit(self.scaler.transform(X), y)
        self.coef_ = pd.Series(self.m.coef_, index=X.columns)
        return self

    def predict(self, d: pd.DataFrame) -> pd.Series:
        return pd.Series(self.m.predict(self.scaler.transform(self._X(d))), index=d.index)


class LGBMModel:
    """Rung 3: shallow LightGBM, L1 (MAE) objective, fixed number of rounds from tuning."""
    name = "lgbm"

    def __init__(self, features: list[str], params: dict, num_boost_round: int):
        self.features, self.params, self.rounds = features, params, num_boost_round

    def fit(self, d: pd.DataFrame, valid: pd.DataFrame | None = None, early_stopping: int | None = None):
        y, self.clip = make_target(d)
        p = {"objective": "l1", "verbosity": -1, "seed": SEED, "deterministic": True, "num_threads": 4, **self.params}
        dtr = lgb.Dataset(d[self.features], y, free_raw_data=False)
        cb, sets = [], []
        if valid is not None:
            yv, _ = make_target(valid, self.clip)
            sets = [lgb.Dataset(valid[self.features], yv, reference=dtr)]
            if early_stopping:
                cb = [lgb.early_stopping(early_stopping, verbose=False)]
        self.m = lgb.train(p, dtr, num_boost_round=self.rounds, valid_sets=sets, callbacks=cb)
        self.best_iteration = self.m.best_iteration or self.rounds
        return self

    def predict(self, d: pd.DataFrame) -> pd.Series:
        return pd.Series(self.m.predict(d[self.features], num_iteration=self.best_iteration), index=d.index)


def xs_z(s: pd.Series) -> pd.Series:
    g = s.groupby(level="date")
    return (s - g.transform("mean")) / g.transform("std")


def ensemble(preds: dict[str, pd.Series]) -> pd.Series:
    """Rung 5: average of per-date cross-sectional z-scores."""
    return pd.concat([xs_z(p) for p in preds.values()], axis=1).mean(axis=1)
