"""One on-disk format for every time series Heatwave uses.

A series is a pandas DataFrame with a tz-aware UTC DatetimeIndex named
``time`` at a regular frequency, saved as ``<name>.parquet`` with a sidecar
``<name>.meta.json``:

{
  "name": "caiso_np15_lmp_dam",
  "freq": "1h",                      # pandas offset alias
  "columns": {"lmp_usd_per_mwh": {"unit": "USD/MWh", "description": "..."}},
  "source": {"title": "...", "url": "...", "license": "...",
             "retrieved_utc": "...", "raw_sha256": {"file": "sha"}},
  "data_kind": "public" | "measured" | "simulated",
  "producer": "heatwave/data/public/caiso.py"
}

Missing values stay NaN (Timestral reads NaN as missing). Processed files
live in data/processed/; raw downloads in data/raw/ (gitignored; the fetch
script and the raw sha256 are committed so anyone can re-fetch and check).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_KINDS = ("public", "measured", "simulated")


def save_series(df: pd.DataFrame, name: str, meta: Mapping[str, Any], root: Path = PROCESSED) -> Path:
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("index must be a tz-aware DatetimeIndex (UTC)")
    if meta.get("data_kind") not in DATA_KINDS:
        raise ValueError(f"meta.data_kind must be one of {DATA_KINDS}")
    for key in ("freq", "columns", "source", "producer"):
        if key not in meta:
            raise ValueError(f"meta missing {key!r}")
    missing = set(df.columns) - set(meta["columns"])
    if missing:
        raise ValueError(f"meta.columns lacks units for {sorted(missing)}")
    df = df.tz_convert("UTC").sort_index()
    df.index.name = "time"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.parquet"
    df.to_parquet(path)
    (root / f"{name}.meta.json").write_text(json.dumps({"name": name, **meta}, indent=1) + "\n")
    return path


def load_series(name: str, root: Path = PROCESSED) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(root / f"{name}.parquet")
    if isinstance(df.index, pd.DatetimeIndex):
        # Files written by pandas 3 default to microsecond resolution; use ns everywhere.
        df.index = df.index.as_unit("ns")
    meta = json.loads((root / f"{name}.meta.json").read_text())
    return df, meta


def list_series(root: Path = PROCESSED) -> list[str]:
    return sorted(p.name[: -len(".meta.json")] for p in root.glob("*.meta.json"))
