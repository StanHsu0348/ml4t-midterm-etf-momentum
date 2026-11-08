"""Project paths and the frozen setup contract."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
ART = RESULTS / "artifacts"
for _p in (RAW, FIG, TAB, ART):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42


@lru_cache
def setup() -> dict:
    with open(ROOT / "setup.yaml") as f:
        return yaml.safe_load(f)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def dev_end() -> pd.Timestamp:
    """Last calendar day of the development sample (day before holdout start)."""
    return ts(setup()["evaluation"]["holdout"][0]) - pd.Timedelta(days=1)


def holdout_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = setup()["evaluation"]["holdout"]
    return ts(a), ts(b)
