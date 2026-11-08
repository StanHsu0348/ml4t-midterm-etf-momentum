"""Research ledger (Ch. 6/16): Strategy -> Trial family -> Trial -> Run.

Every adaptive choice that is evaluated on data is appended here as a run.
Bug fixes are not trials. The trial count feeds the Deflated Sharpe Ratio.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from .config import ROOT, setup

LEDGER = ROOT / "run_log.csv"
FIELDS = [
    "run_id", "logged_at", "strategy", "setup_version", "trial_family", "trial", "run",
    "git_commit", "data_snapshot", "config", "eval_window", "metrics", "decision", "gate",
]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src", "notebooks", "setup.yaml"], cwd=ROOT)
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "no-git"


def file_sha(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def data_snapshot() -> str:
    from .data import PANEL_PATH
    return file_sha(PANEL_PATH) if PANEL_PATH.exists() else "none"


def init_ledger() -> None:
    if not LEDGER.exists():
        with open(LEDGER, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)


def _next_id() -> int:
    init_ledger()
    with open(LEDGER) as f:
        return sum(1 for _ in f)  # header counts as row 0 -> ids start at 1


def log_run(trial_family: str, trial: str, run: str, config: dict, eval_window: str,
            metrics: dict, decision: str = "", gate: str = "") -> int:
    rid = _next_id()
    s = setup()
    row = {
        "run_id": rid,
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": s["strategy_id"],
        "setup_version": s["setup_version"],
        "trial_family": trial_family,
        "trial": trial,
        "run": run,
        "git_commit": git_commit(),
        "data_snapshot": data_snapshot(),
        "config": json.dumps(config, default=str, sort_keys=True),
        "eval_window": eval_window,
        "metrics": json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}, default=str),
        "decision": decision,
        "gate": gate,
    }
    with open(LEDGER, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    return rid


def trial_count(families: list[str] | None = None) -> int:
    import pandas as pd
    if not LEDGER.exists():
        return 0
    df = pd.read_csv(LEDGER)
    if families:
        df = df[df.trial_family.isin(families)]
    return int(df.drop_duplicates(["trial_family", "trial"]).shape[0])
