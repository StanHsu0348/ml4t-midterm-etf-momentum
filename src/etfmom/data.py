"""Data download, storage and point-in-time access with a sealed holdout.

Regenerate the dataset exactly:  python scripts/download_data.py
Storage: data/raw/etf_panel.parquet (long format, one row per symbol-session)
         data/raw/etf_actions.parquet (dividends / splits with ex-dates)
         data/raw/metadata.json       (vendor, download time, versions, checksum)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ART, DATA, RAW, ROOT, dev_end, holdout_window, setup

PANEL_PATH = RAW / "etf_panel.parquet"
ACTIONS_PATH = RAW / "etf_actions.parquet"
META_PATH = RAW / "metadata.json"
FROZEN_SPEC = ART / "frozen_spec.json"
HOLDOUT_LOG = ROOT / "results" / "holdout_access.log"


def universe() -> pd.DataFrame:
    return pd.read_csv(DATA / "universe.csv")


# ----------------------------------------------------------------------------- download
def download(symbols: list[str] | None = None) -> pd.DataFrame:
    import yfinance as yf

    s = setup()["data"]
    syms = symbols or universe().symbol.tolist()
    end_excl = (pd.Timestamp(s["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames, actions = [], []
    for i in range(0, len(syms), 20):
        batch = syms[i:i + 20]
        raw = yf.download(batch, start=s["start"], end=end_excl, auto_adjust=False, actions=True,
                          group_by="ticker", progress=False, threads=True)
        for sym in batch:
            if sym not in raw.columns.get_level_values(0):
                continue
            d = raw[sym].dropna(how="all").copy()
            d.columns = [c.lower().replace(" ", "_") for c in d.columns]
            d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
            d["symbol"] = sym
            for col, kind in (("dividends", "dividend"), ("stock_splits", "split")):
                if col in d:
                    a = d.loc[d[col].fillna(0) != 0, [col]].rename(columns={col: "value"})
                    a["symbol"], a["kind"] = sym, kind
                    actions.append(a.reset_index(names="date"))
            frames.append(d.reset_index(names="date"))
    df = pd.concat(frames, ignore_index=True)
    # adjusted OHLC = raw OHLC * (adj_close / close): split + distribution adjusted
    factor = df["adj_close"] / df["close"]
    out = pd.DataFrame({
        "date": df["date"], "symbol": df["symbol"],
        "open": df["open"] * factor, "high": df["high"] * factor,
        "low": df["low"] * factor, "close": df["adj_close"],
        "open_raw": df["open"], "close_raw": df["close"],
        "volume": df["volume"].astype("float64"), "adj_factor": factor,
    }).sort_values(["symbol", "date"]).reset_index(drop=True)
    out.to_parquet(PANEL_PATH, index=False)
    act = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame(columns=["date", "value", "symbol", "kind"])
    act.to_parquet(ACTIONS_PATH, index=False)
    from .ledger import file_sha
    meta = {
        "vendor": "Yahoo Finance via yfinance", "yfinance_version": yf.__version__,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": {"start": s["start"], "end_inclusive": s["end"], "auto_adjust": False},
        "timestamp_semantics": "date = NYSE session date; open/close = official session open/close (09:30/16:00 ET)",
        "adjustment": "adjusted = raw * adj_close/close (Yahoo split + dividend back-adjustment)",
        "rows": int(len(out)), "symbols": int(out.symbol.nunique()),
        "sha256_12": file_sha(PANEL_PATH),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    return out


# ----------------------------------------------------------------------------- access
def _read() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError("Run `python scripts/download_data.py` first.")
    df = pd.read_parquet(PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_panel() -> pd.DataFrame:
    """Development data ONLY: every row with date >= holdout start is removed."""
    df = _read()
    return df[df.date <= dev_end()].reset_index(drop=True)


def load_panel_with_holdout(reason: str, allow_rerun: bool = False) -> pd.DataFrame:
    """The single gate to the sealed holdout.

    Requires a frozen specification (written by notebooks/06_backtest_risk) and
    logs every access. A second access raises unless allow_rerun=True, which is
    reserved for the reproducibility re-run and is itself logged.
    """
    if not FROZEN_SPEC.exists():
        raise PermissionError("Holdout is sealed: freeze the specification first.")
    HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    prior = HOLDOUT_LOG.read_text().strip().splitlines() if HOLDOUT_LOG.exists() else []
    if prior and not allow_rerun:
        raise PermissionError(f"Holdout already touched: {prior[0]}")
    spec = json.loads(FROZEN_SPEC.read_text())
    with open(HOLDOUT_LOG, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | touch #{len(prior) + 1} | "
                f"spec={spec.get('spec_hash')} | rerun={allow_rerun} | {reason}\n")
    df = _read()
    return df[df.date <= holdout_window()[1]].reset_index(drop=True)


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="symbol", values=col).sort_index()


def month_end_sessions(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby([dates.year, dates.month]).max().values)


def eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean (date x symbol): eligible at close of each date using data <= date."""
    u = setup()["universe"]["eligibility"]
    close = wide(df, "close")
    dv = wide(df, "close_raw") * wide(df, "volume")
    hist = close.notna().cumsum()
    adv = dv.rolling(u["adv_window_days"], min_periods=u["adv_window_days"] // 2).median()
    return (hist >= u["min_history_days"]) & (adv >= u["min_median_dollar_volume_usd"]) & close.notna()


def half_spread_usd(symbol: str) -> float:
    c = setup()["costs"]["half_spread_usd"]
    if symbol in {"SPY", "QQQ", "IWM", "EFA", "EEM", "DIA", "VTI"}:
        return c["mega"]
    if symbol.startswith("XL"):
        return c["sector_spdr"]
    return c["default"]


def per_leg_cost_bps(df: pd.DataFrame) -> pd.DataFrame:
    """date x symbol per-leg cost (fraction) at the raw open price of that session."""
    c = setup()["costs"]
    open_raw = wide(df, "open_raw")
    hs = pd.Series({s: half_spread_usd(s) for s in open_raw.columns})
    return (hs + c["commission_per_share_usd"]) / open_raw


__all__ = [n for n in dir() if not n.startswith("_")] + ["np"]
