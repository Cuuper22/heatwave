#!/usr/bin/env python
"""Collect German (DE-LU) day-ahead prices, public net generation and day-ahead forecasts from Energy-Charts.

Source: Fraunhofer ISE Energy-Charts API, https://api.energy-charts.info/ ; the API states the data
licence per response: "CC BY 4.0 (creativecommons.org/licenses/by/4.0) from Bundesnetzagentur | SMARD.de".

Raw   : data/raw/energy_charts/<endpoint>_<start>.json (gitignored); sha256 in data/processed/public_energy_charts/raw_manifest.json
Output: data/processed/public_de_price_dam_1h.*        hourly mean of the day-ahead price (EUR/MWh)
        data/processed/public_de_price_dam_native.*    day-ahead price at its native step (60 min, then 15 min)
        data/processed/public_de_generation.*          15 min net generation by type (MW), load, and a lifecycle intensity proxy
        data/processed/public_de_forecast_dayahead.*   15 min day-ahead forecasts of load, solar, wind onshore, wind offshore
Usage : python scripts/fetch_energy_charts.py [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from heatwave.data.public import energy_charts as ec
from heatwave.data.public._common import PROCESSED_ROOT, RAW_ROOT, Blocked, Manifest, http_get, utc_now
from heatwave.data.schema import save_series

RAW = RAW_ROOT / "energy_charts"
MAN = PROCESSED_ROOT / "public_energy_charts"
API = "https://api.energy-charts.info"
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(days=90)
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), from Bundesnetzagentur | SMARD.de (as stated in the API response)"
ENDPOINTS = {
    "price": ("/price", {"bzn": "DE-LU"}),
    "power": ("/public_power", {"country": "de"}),
    "fc_load": ("/public_power_forecast", {"country": "de", "production_type": "load", "forecast_type": "day-ahead"}),
    "fc_solar": ("/public_power_forecast", {"country": "de", "production_type": "solar", "forecast_type": "day-ahead"}),
    "fc_wind_onshore": ("/public_power_forecast", {"country": "de", "production_type": "wind_onshore", "forecast_type": "day-ahead"}),
    "fc_wind_offshore": ("/public_power_forecast", {"country": "de", "production_type": "wind_offshore", "forecast_type": "day-ahead"}),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    man = Manifest(MAN / "raw_manifest.json")
    data: dict[str, list[dict]] = {k: [] for k in ENDPOINTS}
    shas: dict[str, dict[str, str]] = {k: {} for k in ENDPOINTS}
    for key, (path, params) in ENDPOINTS.items():
        s = START
        while s < end:
            e = min(s + STEP, end)
            rel = f"{key}_{s:%Y%m%d}.json"
            dest = RAW / rel
            final = e <= datetime.now(timezone.utc) - timedelta(days=3)
            if not (dest.exists() and final):
                q = {**params, "start": f"{s:%Y-%m-%dT%H:%M}Z", "end": f"{e:%Y-%m-%dT%H:%M}Z"}
                try:
                    r = http_get(API + path, params=q, retries=6, backoff=10.0)
                except Blocked as exc:
                    print("blocked:", exc)
                    s = e
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                man.record(rel, dest, r.url)
                print(rel, len(r.content), flush=True)
                time.sleep(2.0)
            else:
                man.record(rel, dest)
            shas[key][rel] = man.sha(rel)
            data[key].append(json.loads(dest.read_text()))
            s = e
    man.save()
    src = {"url": API, "license": LICENSE, "retrieved_utc": utc_now()}

    # day-ahead price
    price = ec.price_frame(data["price"])
    save_series(price["native"], "public_de_price_dam_native", {
        "freq": price["native_freq"],
        "columns": {"price_eur_per_mwh": {"unit": "EUR/MWh", "description": "DE-LU day-ahead auction price at its native step"}},
        "source": {"title": "Energy-Charts /price?bzn=DE-LU (day-ahead)", **src, "raw_sha256": shas["price"]},
        "data_kind": "public", "producer": "scripts/fetch_energy_charts.py",
        "notes": f"Native step changes from 60 min to 15 min on {price['switch_utc']}; this file covers only the 15-min period."})
    save_series(price["hourly"], "public_de_price_dam_1h", {
        "freq": "1h",
        "columns": {"price_eur_per_mwh": {"unit": "EUR/MWh", "description": "DE-LU day-ahead price; hourly mean of the native values (identity while the native step is 60 min)"}},
        "source": {"title": "Energy-Charts /price?bzn=DE-LU (day-ahead)", **src, "raw_sha256": shas["price"]},
        "data_kind": "public", "producer": "scripts/fetch_energy_charts.py",
        "notes": "Day-ahead prices for day D are published around 12:45 CET on D-1: known-future covariate."})

    # generation + intensity proxy
    gen = ec.power_frame(data["power"])
    cols = {c: {"unit": "MW" if not c.startswith(("renewable_share", "intensity")) else ("%" if c.startswith("renewable_share") else "gCO2eq/kWh"),
                "description": ec.GEN_DESC.get(c, c)} for c in gen.columns}
    save_series(gen, "public_de_generation", {
        "freq": "15min", "columns": cols,
        "source": {"title": "Energy-Charts /public_power?country=de (public net electricity generation)", **src, "raw_sha256": shas["power"]},
        "data_kind": "public", "producer": "scripts/fetch_energy_charts.py",
        "emission_factors": {"source": ec.FACTOR_SOURCE, "gco2eq_per_kwh": ec.FACTORS, "mapping": ec.FACTOR_MAP},
        "notes": "intensity_lifecycle_proxy_g_per_kwh is production-based (no imports), computed here from IPCC AR5 lifecycle medians; "
                 "generation types without a factor are excluded and reported as intensity_factor_coverage_pct."})

    # day-ahead forecasts
    fc = ec.forecast_frame({k: data[k] for k in data if k.startswith("fc_")})
    save_series(fc, "public_de_forecast_dayahead", {
        "freq": "15min",
        "columns": {c: {"unit": "MW", "description": f"Energy-Charts day-ahead forecast: {c.replace('_forecast_mw', '').replace('_', ' ')}"} for c in fc.columns},
        "source": {"title": "Energy-Charts /public_power_forecast (day-ahead)", **src, "raw_sha256": {k: v for kk in shas if kk.startswith("fc_") for k, v in shas[kk].items()}},
        "data_kind": "public", "producer": "scripts/fetch_energy_charts.py",
        "notes": "Archived day-ahead forecasts: known-future covariates for load and renewable output."})
    print("saved", len(price["hourly"]), len(gen), len(fc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
