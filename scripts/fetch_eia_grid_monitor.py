#!/usr/bin/env python
"""Collect EIA Hourly Electric Grid Monitor (Form EIA-930) series for three US balancing authorities from the bulk file.

Source: U.S. Energy Information Administration, bulk file https://www.eia.gov/opendata/bulk/EBA.zip
(api.eia.gov is blocked from this VM; www.eia.gov is reachable). EIA data are U.S. government works; EIA asks for
citation of the source (https://www.eia.gov/about/copyright.php). Hourly values in UTC ("H" series).

Balancing authorities: CISO (California ISO), ERCO (ERCOT, Texas), PJM (PJM Interconnection, mid-Atlantic incl. northern Virginia).
Series per BA: D demand, DF day-ahead demand forecast (published by the BA before the operating day), NG net generation,
TI total interchange, NG.<fuel> net generation by energy source (COL coal, NG natural gas, NUC, OIL, OTH, SUN, WAT, WND).

Raw   : data/raw/eia/EBA.zip (gitignored, ~0.7 GB); sha256 in data/processed/public_eia/raw_manifest.json
Output: data/processed/public_eia_<ba>_hourly.*  (hourly UTC, 2024-01-01 ..)
Usage : python scripts/fetch_eia_grid_monitor.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile

import numpy as np
import pandas as pd

from heatwave.data.public import energy_charts as ec
from heatwave.data.public._common import PROCESSED_ROOT, RAW_ROOT, Manifest, download, utc_now
from heatwave.data.schema import save_series

RAW = RAW_ROOT / "eia"
MAN = PROCESSED_ROOT / "public_eia"
URL = "https://www.eia.gov/opendata/bulk/EBA.zip"
BAS = {"CISO": "California ISO", "ERCO": "ERCOT (Texas)", "PJM": "PJM Interconnection"}
FUELS = {"COL": "coal", "NG": "natural_gas", "NUC": "nuclear", "OIL": "oil", "OTH": "other", "SUN": "solar", "WAT": "hydro", "WND": "wind"}
# IPCC AR5 lifecycle medians (see heatwave/data/public/energy_charts.py); EIA categories without a factor are excluded.
NUC_FACTOR = 12  # IPCC AR5 Annex III Table A.III.2, nuclear lifecycle median
START = pd.Timestamp("2024-01-01", tz="UTC")


def wanted() -> dict[str, str]:
    out = {}
    for ba in BAS:
        for code, col in {"D": "demand_mw", "DF": "demand_forecast_day_ahead_mw", "NG": "net_generation_mw", "TI": "interchange_mw"}.items():
            out[f"EBA.{ba}-ALL.{code}.H"] = (ba, col)
        for f, name in FUELS.items():
            out[f"EBA.{ba}-ALL.NG.{f}.H"] = (ba, f"ng_{name}_mw")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "EBA.zip"
    if a.refresh and zpath.exists():
        zpath.unlink()
    download(URL, zpath)
    man = Manifest(MAN / "raw_manifest.json")
    man.record("EBA.zip", zpath, URL)
    man.save()
    want = wanted()
    got: dict[str, dict[str, pd.Series]] = {ba: {} for ba in BAS}
    meta_names: dict[str, str] = {}
    pat = re.compile(rb'^\{"series_id":"([^"]+)"')
    with zipfile.ZipFile(zpath) as z, z.open("EBA.txt") as fh:
        for line in fh:
            m = pat.match(line)
            if not m:
                continue
            sid = m.group(1).decode()
            if sid not in want:
                continue
            j = json.loads(line)
            ba, col = want[sid]
            t = pd.to_datetime([r[0] for r in j["data"]], format="%Y%m%dT%H", utc=True)
            v = pd.to_numeric(pd.Series([r[1] for r in j["data"]], dtype="object"), errors="coerce").to_numpy()
            s = pd.Series(v, index=t).sort_index()
            got[ba][col] = s[s.index >= START]
            meta_names[sid] = j["name"]
    for ba, cols in got.items():
        df = pd.DataFrame(cols)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC", name="time")).astype("float64")
        # lifecycle intensity proxy from net generation by source (production-based)
        fac = {"ng_coal_mw": ec.FACTORS["coal"], "ng_natural_gas_mw": ec.FACTORS["gas_cc"], "ng_nuclear_mw": NUC_FACTOR,
               "ng_solar_mw": ec.FACTORS["solar_pv_utility"], "ng_hydro_mw": ec.FACTORS["hydro"], "ng_wind_mw": ec.FACTORS["wind_onshore"]}
        use = [c for c in fac if c in df.columns]
        e = df[use].clip(lower=0)
        covered = e.sum(axis=1, min_count=len(use))
        other = [c for c in df.columns if c.startswith("ng_") and c not in fac]
        total = covered + df[other].clip(lower=0).sum(axis=1, min_count=1).fillna(0)
        df["intensity_lifecycle_proxy_g_per_kwh"] = (e * pd.Series(fac)[use]).sum(axis=1, min_count=len(use)) / covered.where(covered > 0)
        df["intensity_factor_coverage_pct"] = 100.0 * covered / total.where(total > 0)
        cmeta = {c: {"unit": "MW", "description": f"EIA-930 hourly {c.replace('_', ' ')} (megawatthours per hour = average MW)"} for c in df.columns
                 if c.endswith("_mw")}
        cmeta["intensity_lifecycle_proxy_g_per_kwh"] = {"unit": "gCO2eq/kWh", "description": "production-weighted IPCC AR5 lifecycle medians over coal, gas, nuclear, solar, hydro, wind; oil and other excluded"}
        cmeta["intensity_factor_coverage_pct"] = {"unit": "%", "description": "share of positive net generation covered by a factor"}
        save_series(df, f"public_eia_{ba.lower()}_hourly", {
            "freq": "1h", "columns": cmeta,
            "source": {"title": f"EIA Hourly Electric Grid Monitor (EIA-930), {BAS[ba]} ({ba}), bulk file EBA.zip", "url": URL,
                       "license": "U.S. government work; cite EIA (https://www.eia.gov/about/copyright.php)",
                       "retrieved_utc": man.data["EBA.zip"]["retrieved_utc"], "raw_sha256": {"EBA.zip": man.sha("EBA.zip")},
                       "series_ids": sorted(k for k, (b, _) in want.items() if b == ba and k in meta_names)},
            "data_kind": "public", "producer": "scripts/fetch_eia_grid_monitor.py",
            "emission_factors": {"source": ec.FACTOR_SOURCE, "nuclear_gco2eq_per_kwh": NUC_FACTOR},
            "notes": "demand_forecast_day_ahead_mw is the balancing authority's forecast submitted before the operating day: a known-future covariate. "
                     "BA-reported values can contain outliers and gaps (EIA does not clean them in the bulk file)."})
        print(f"saved {ba}: {len(df)} rows {df.index[0]} .. {df.index[-1]}; columns {len(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
