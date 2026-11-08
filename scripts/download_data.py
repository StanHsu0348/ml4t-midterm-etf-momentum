"""Regenerate the dataset exactly: `uv run python scripts/download_data.py` (from repo root)."""
import json

from etfmom.data import META_PATH, download

if __name__ == "__main__":
    df = download()
    print(json.dumps(json.loads(META_PATH.read_text()), indent=2))
