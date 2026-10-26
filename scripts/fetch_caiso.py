#!/usr/bin/env python
"""Collect CAISO day-ahead / real-time prices (OASIS) and Today's Outlook fuel mix, CO2 and demand.

Raw   : data/raw/caiso/oasis/<market>_<start>.zip, data/raw/caiso/outlook/<day>_<name>.csv (gitignored)
Output: data/processed/public_caiso_np15_lmp_dam.*   hourly day-ahead LMP, TH_NP15_GEN-APND
        data/processed/public_caiso_np15_lmp_rtm.*   5-min real-time LMP, same node
        data/processed/public_caiso_fuelmix.*        5-min MW by fuel (Outlook)
        data/processed/public_caiso_carbon.*         5-min CO2 by resource, supply and computed intensity
        data/processed/public_caiso_demand.*         5-min demand with CAISO day-ahead and hour-ahead forecast
        data/processed/public_caiso/raw_manifest.json  sha256 of every raw file

Re-runnable: finished chunks/days (older than 3 days) already on disk are not requested again.
OASIS is asked one request per 6 s (limit: at most 31 days per request).
Usage: python scripts/fetch_caiso.py [--skip-oasis] [--skip-outlook] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from heatwave.data.public import caiso
from heatwave.data.public._common import (
    PROCESSED_ROOT, RAW_ROOT, Blocked, Manifest, http_get, utc_now, write_json,
)
from heatwave.data.schema import save_series

RAW = RAW_ROOT / "caiso"
MAN_DIR = PROCESSED_ROOT / "public_caiso"
OASIS_PAGE = "https://oasis.caiso.com/"
LICENSE = "CAISO public market data; no explicit open license found. Cited with source; check CAISO terms before redistribution."


def fetch_oasis(manifest: Manifest, end: datetime) -> list[str]:
    errors = []
    jobs = [("dam", "PRC_LMP", "DAM", "12"), ("rtm", "PRC_INTVL_LMP", "RTM", "3")]
    for tag, query, market, version in jobs:
        for s, e in caiso.chunk_starts(end):
            rel = f"oasis/{tag}_{s:%Y%m%d}.zip"
            dest = RAW / rel
            final = e <= datetime.now(timezone.utc) - timedelta(days=3)
            if dest.exists() and final:
                manifest.record(rel, dest)
                continue
            params = caiso.oasis_params(query, market, version, s, e)
            try:
                r = http_get(caiso.OASIS_URL, params=params, retries=8, backoff=15.0)
            except Blocked as exc:
                errors.append(str(exc))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            manifest.record(rel, dest, r.url)
            manifest.data[rel]["chunk_end_utc"] = e.isoformat()
            print(f"{rel}: {len(r.content)} bytes", flush=True)
            manifest.save()
            time.sleep(caiso.OASIS_SLEEP_S)
    return errors


def fetch_outlook(manifest: Manifest, end: datetime, workers: int = 3) -> list[str]:
    days = pd.date_range(caiso.START.date(), (end - timedelta(days=1)).date(), freq="1D")
    todo = []
    for d in days:
        for name in ("fuelsource", "co2", "demand"):
            rel = f"outlook/{d:%Y%m%d}_{name}.csv"
            dest = RAW / rel
            final = d <= pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=3)
            if dest.exists() and dest.stat().st_size > 0 and final:
                continue
            todo.append((d, name, rel))
    print(f"outlook: {len(todo)} files to request", flush=True)
    errors: list[str] = []

    def one(item):
        d, name, rel = item
        url = caiso.OUTLOOK_URL.format(day=f"{d:%Y%m%d}", name=name)
        try:
            r = http_get(url, retries=5, backoff=5.0)
        except Blocked as exc:
            return str(exc)
        except Exception as exc:  # 404 for a missing day etc.
            return f"{url}: {exc}"
        dest = RAW / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        time.sleep(0.4)
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, err in enumerate(ex.map(one, todo)):
            if err:
                errors.append(err)
            if i % 200 == 0:
                print(f"  outlook {i}/{len(todo)}", flush=True)
    for d in days:
        for name in ("fuelsource", "co2", "demand"):
            rel = f"outlook/{d:%Y%m%d}_{name}.csv"
            if (RAW / rel).exists():
                manifest.record(rel, RAW / rel)
    return errors


def process_oasis(manifest: Manifest) -> None:
    for tag, freq, name, title in [
        ("dam", "1h", "public_caiso_np15_lmp_dam", "CAISO OASIS PRC_LMP day-ahead market (DAM), node TH_NP15_GEN-APND"),
        ("rtm", "5min", "public_caiso_np15_lmp_rtm", "CAISO OASIS PRC_INTVL_LMP real-time market (RTM) 5-min LMP, node TH_NP15_GEN-APND"),
    ]:
        parts, shas = [], {}
        for p in sorted((RAW / "oasis").glob(f"{tag}_*.zip")):
            w = caiso.oasis_to_wide(caiso.read_oasis_zip(p.read_bytes()), freq)
            if not w.empty:
                parts.append(w)
            shas[f"oasis/{p.name}"] = manifest.sha(f"oasis/{p.name}")
        df = caiso.regularize(pd.concat(parts), freq)
        cols = {v[0]: {"unit": "USD/MWh", "description": f"{v[1]} at {caiso.NODE}"} for k, v in caiso.LMP_TYPES.items() if v[0] in df.columns}
        df = df[list(cols)]
        meta = {
            "freq": freq, "columns": cols,
            "source": {"title": title, "url": caiso.OASIS_URL, "license": LICENSE, "retrieved_utc": utc_now(), "raw_sha256": shas},
            "data_kind": "public", "producer": "scripts/fetch_caiso.py",
            "notes": ("Day-ahead prices for operating day D are published about 1 pm Pacific on D-1, so they are known when D starts: "
                       "usable as a known-future covariate for real-time price and load. Timestamps are interval START in UTC.")
            if tag == "dam" else "Real-time 5-minute LMP; interval START in UTC. NaN where OASIS returned no row.",
        }
        save_series(df, name, meta)
        print(f"saved {name}: {len(df)} rows {df.index[0]} .. {df.index[-1]}")


def process_outlook(manifest: Manifest) -> None:
    frames = {"fuelsource": [], "co2": [], "demand": []}
    shas: dict[str, dict[str, str]] = {k: {} for k in frames}
    for p in sorted((RAW / "outlook").glob("*.csv")):
        day, name = p.stem.split("_", 1)
        try:
            d = caiso.parse_outlook_day(p.read_text(), day)
        except Exception as exc:
            print(f"skip {p.name}: {exc}")
            continue
        if not d.empty:
            frames[name].append(d)
        shas[name][f"outlook/{p.name}"] = manifest.sha(f"outlook/{p.name}")
    fuel = caiso.regularize(pd.concat(frames["fuelsource"]), "5min")
    co2 = caiso.regularize(pd.concat(frames["co2"]), "5min")
    dem = caiso.regularize(pd.concat(frames["demand"]), "5min")
    src = {"url": "https://www.caiso.com/outlook/history/", "license": LICENSE, "retrieved_utc": utc_now()}
    fuel.columns = [c.lower().replace(" ", "_") + "_mw" for c in fuel.columns]
    save_series(fuel, "public_caiso_fuelmix", {
        "freq": "5min", "columns": {c: {"unit": "MW", "description": "CAISO Today's Outlook fuel source output (batteries negative when charging; imports included)"} for c in fuel.columns},
        "source": {"title": "CAISO Today's Outlook history: fuelsource.csv", **src, "raw_sha256": shas["fuelsource"]},
        "data_kind": "public", "producer": "scripts/fetch_caiso.py",
        "notes": "Pacific wall-clock times in the raw files were converted to UTC (DST handled)."})
    carbon = caiso.carbon_from_outlook(fuel.rename(columns=lambda c: c), co2)
    cmeta = {c: {"unit": "mTCO2/h", "description": "CAISO-published CO2 emission rate of one resource group (unit as printed on https://www.caiso.com/todays-outlook/emissions)"} for c in carbon.columns if c.startswith("co2_")}
    cmeta["co2_total_mt_per_h"] = {"unit": "mTCO2/h", "description": "sum of the CAISO CO2 resource columns"}
    cmeta["supply_mw"] = {"unit": "MW", "description": "sum of all fuelsource columns (net of battery charging)"}
    cmeta["intensity_kg_per_mwh"] = {"unit": "kgCO2/MWh", "description": "1000 * co2_total_mt_per_h / supply_mw; computed here, not published by CAISO"}
    cmeta["in_state_generation_mw"] = {"unit": "MW", "description": "sum of the fuelsource columns except imports, batteries clipped at zero"}
    cmeta["intensity_in_state_kg_per_mwh"] = {"unit": "kgCO2/MWh", "description": "1000 * (CO2 of non-import resources) / in_state_generation_mw; computed here, never negative"}
    cmeta["intensity_kg_per_mwh"]["description"] += "; negative while CAISO's published imports CO2 is negative (net export)"
    save_series(carbon, "public_caiso_carbon", {
        "freq": "5min", "columns": cmeta,
        "source": {"title": "CAISO Today's Outlook history: co2.csv and fuelsource.csv", **src, "raw_sha256": {**shas["co2"], **shas["fuelsource"]}},
        "data_kind": "public", "producer": "scripts/fetch_caiso.py",
        "notes": "Intensity is a derived quantity (see column description); the import emission factor inside CAISO's co2.csv is CAISO's own."})
    dem.columns = ["demand_forecast_day_ahead_mw", "demand_forecast_hour_ahead_mw", "demand_mw", "demand_response_mw"][: dem.shape[1]]
    save_series(dem, "public_caiso_demand", {
        "freq": "5min", "columns": {c: {"unit": "MW", "description": c.replace("_", " ")} for c in dem.columns},
        "source": {"title": "CAISO Today's Outlook history: demand.csv", **src, "raw_sha256": shas["demand"]},
        "data_kind": "public", "producer": "scripts/fetch_caiso.py",
        "notes": "demand_forecast_day_ahead_mw is the CAISO day-ahead forecast published before the operating day: a known-future covariate."})
    print(f"saved outlook series: fuel {len(fuel)}, carbon {len(carbon)}, demand {len(dem)} rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-oasis", action="store_true")
    ap.add_argument("--skip-outlook", action="store_true")
    ap.add_argument("--no-process", action="store_true")
    ap.add_argument("--end", default=None, help="exclusive end day YYYY-MM-DD (default: today UTC)")
    a = ap.parse_args()
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    manifest = Manifest(MAN_DIR / "raw_manifest.json")
    errors: list[str] = []
    if not a.skip_oasis:
        errors += fetch_oasis(manifest, end)
    manifest.save()
    if not a.skip_outlook:
        errors += fetch_outlook(manifest, end)
    manifest.save()
    if not a.no_process:
        process_oasis(manifest)
        process_outlook(manifest)
    if errors:
        write_json(MAN_DIR / "fetch_errors.json", errors[:200])
        print(f"{len(errors)} request errors (first: {errors[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
