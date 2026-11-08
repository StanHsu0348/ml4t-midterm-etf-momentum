"""Print text outputs of an executed notebook: python scripts/show_nb.py notebooks/03_x.ipynb"""
import sys
import nbformat
nb = nbformat.read(sys.argv[1], 4)
for c in nb.cells:
    for o in c.get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain", "")
        if t:
            print(t[: int(sys.argv[2]) if len(sys.argv) > 2 else 3000]); print("-----")
