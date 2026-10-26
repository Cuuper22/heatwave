#!/usr/bin/env python
"""Collect Great Britain half-hourly carbon intensity (actual and forecast) and generation mix.

Source: National Energy System Operator (formerly National Grid ESO) Carbon Intensity API,
https://carbon-intensity.github.io/api-definitions/ ; data licence CC BY 4.0
(https://github.com/carbon-intensity/api-definitions, see data/SOURCES.md).

Raw   : data/raw/uk_carbon/<endpoint>_<start>.json (gitignored), sha256 in data/processed/public_uk_carbon/raw_manifest.json
Output: data/processed/public_uk_carbon_intensity.*   30 min, actual + forecast gCO2/kWh
        data/processed/public_uk_generation_mix.*      30 min, percent share per fuel
Usage : python scripts/fetch_uk_carbon.py [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from heatwave.data.public._common import PROCESSED_ROOT, RAW_ROOT, Blocked, Manifest, http_get, utc_now
from heatwave.data.schema import save_series

RAW = RAW_ROOT / "uk_carbon"
MAN = PROCESSED_ROOT / "public_uk_carbon"
API = "https://api.carbonintensity.org.uk"
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(days=28)  # API refuses ranges above 31 days
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), data from the National Energy System Operator carbon intensity API"


def chunks(end: datetime):
    s = START
    while s < end:
        e = min(s + STEP, end)
        yield s, e
        s = e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    man = Manifest(MAN / "raw_manifest.json")
    inten, mix = [], []
    shas: dict[str, dict[str, str]] = {"intensity": {}, "generation": {}}
    for ep in ("intensity", "generation"):
        for s, e in chunks(end):
            rel = f"{ep}_{s:%Y%m%d}.json"
            dest = RAW / rel
            final = e <= datetime.now(timezone.utc) - timedelta(days=3)
            if not (dest.exists() and final):
                url = f"{API}/{ep}/{s:%Y-%m-%dT%H:%MZ}/{e:%Y-%m-%dT%H:%MZ}"
                try:
                    r = http_get(url, retries=6, backoff=5.0)
                except Blocked as exc:
                    print("blocked:", exc)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                man.record(rel, dest, url)
                print(rel, len(r.content), flush=True)
                time.sleep(1.0)
            else:
                man.record(rel, dest)
            shas[ep][rel] = man.sha(rel)
            rows = json.loads(dest.read_text()).get("data", [])
            for row in rows:
                t = pd.Timestamp(row["from"])
                if ep == "intensity":
                    inten.append({"time": t, "intensity_actual_g_per_kwh": row["intensity"].get("actual"), "intensity_forecast_g_per_kwh": row["intensity"].get("forecast")})
                else:
                    d = {"time": t}
                    d.update({f"share_{g['fuel']}_pct": g["perc"] for g in row["generationmix"]})
                    mix.append(d)
    man.save()
    src = {"url": API, "license": LICENSE, "retrieved_utc": utc_now()}
    df = pd.DataFrame(inten).drop_duplicates("time", keep="last").set_index("time").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="30min", tz="UTC", name="time")).astype("float64")
    save_series(df, "public_uk_carbon_intensity", {
        "freq": "30min",
        "columns": {
            "intensity_actual_g_per_kwh": {"unit": "gCO2/kWh", "description": "GB national carbon intensity, actual (estimated from metered generation), period start in UTC"},
            "intensity_forecast_g_per_kwh": {"unit": "gCO2/kWh", "description": "forecast value served by the API for that period; the API does not document the forecast horizon/vintage of archived values"},
        },
        "source": {"title": "Carbon Intensity API (GB), /intensity", **src, "raw_sha256": shas["intensity"]},
        "data_kind": "public", "producer": "scripts/fetch_uk_carbon.py",
        "notes": "Forecast is the value the API returns for past periods; treat its vintage as unknown (not a verified day-ahead vintage)."})
    g = pd.DataFrame(mix).drop_duplicates("time", keep="last").set_index("time").sort_index()
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="30min", tz="UTC", name="time")).astype("float64")
    save_series(g, "public_uk_generation_mix", {
        "freq": "30min",
        "columns": {c: {"unit": "%", "description": "share of GB generation from this fuel (percent), as served by the API"} for c in g.columns},
        "source": {"title": "Carbon Intensity API (GB), /generation", **src, "raw_sha256": shas["generation"]},
        "data_kind": "public", "producer": "scripts/fetch_uk_carbon.py"})
    print("saved", len(df), len(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
