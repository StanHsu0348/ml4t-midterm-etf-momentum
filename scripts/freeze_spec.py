"""Freeze the specification before the single holdout run.

Hashes setup.yaml, src/, notebooks 01-05 and the frozen hyper-parameters /
deployed rung, and writes results/artifacts/frozen_spec.json. The holdout gate
(load_panel_with_holdout) refuses to open without this file.
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "setup.yaml", *sorted((ROOT / "src" / "etfmom").glob("*.py")),
         *sorted((ROOT / "notebooks").glob("0[1-5]_*.py")),
         ROOT / "results" / "artifacts" / "frozen_hyperparams.json", ROOT / "results" / "artifacts" / "deployed.json"]

if __name__ == "__main__":
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.relative_to(ROOT).as_posix().encode()); h.update(f.read_bytes())
    spec = {
        "spec_hash": h.hexdigest()[:16],
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": [f.relative_to(ROOT).as_posix() for f in FILES],
        "hyperparams": json.loads((ROOT / "results/artifacts/frozen_hyperparams.json").read_text()),
        "deployed": json.loads((ROOT / "results/artifacts/deployed.json").read_text())["deployed"],
        "confirmatory_test": "deployed rung, EW top-10, assumed costs, no overlay: holdout mean daily IC > 0 AND net Sharpe > 0",
    }
    out = ROOT / "results" / "artifacts" / "frozen_spec.json"
    out.write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec, indent=2))
