"""Convert notebooks/*.py (jupytext percent format) to .ipynb and execute them in order.

Usage: uv run python scripts/run_notebooks.py [01 02 ...]
"""
import sys
from pathlib import Path

import jupytext
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

if __name__ == "__main__":
    wanted = sys.argv[1:]
    for py in sorted(NB.glob("[0-9][0-9]_*.py")):
        if wanted and py.name[:2] not in wanted:
            continue
        print(f"== executing {py.name}", flush=True)
        nb = jupytext.read(py)
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        jupytext.write(nb, py.with_suffix(".ipynb"))
