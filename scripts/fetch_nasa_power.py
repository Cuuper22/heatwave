#!/usr/bin/env python
"""Collect hourly weather actuals (reanalysis/assimilation, not forecasts) from NASA POWER for three points.

Source: NASA Langley Research Center POWER Project, Hourly API v2 ("data obtained from the POWER Project's
Hourly 2.x.x version"), https://power.larc.nasa.gov/docs/services/api/ ; referencing terms:
https://power.larc.nasa.gov/docs/referencing/ (cite the service, version and access date; no licence fee).

Raw   : data/raw/nasa_power/<region>_<year>.json (gitignored); sha256 in data/processed/public_nasa_power/raw_manifest.json
Output: data/processed/public_weather_nasa_<region>.*   hourly UTC, 2024-01-01 .. latest available
Usage : python scripts/fetch_nasa_power.py [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from heatwave.data.public._common import PROCESSED_ROOT, RAW_ROOT, Blocked, Manifest, http_get, utc_now
from heatwave.data.public.weather import LOCATIONS
from heatwave.data.schema import save_series

RAW = RAW_ROOT / "nasa_power"
MAN = PROCESSED_ROOT / "public_nasa_power"
API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMS = {
    "T2M": ("temperature_2m_c", "degC", "air temperature at 2 m"),
    "RH2M": ("relative_humidity_2m_pct", "%", "relative humidity at 2 m"),
    "T2MDEW": ("dew_point_2m_c", "degC", "dew/frost point at 2 m"),
    "T2MWET": ("wet_bulb_2m_c", "degC", "wet-bulb temperature at 2 m (computed by NASA POWER)"),
    "WS10M": ("wind_speed_10m_mps", "m/s", "wind speed at 10 m"),
    "ALLSKY_SFC_SW_DWN": ("shortwave_radiation_wh_m2", "Wh/m2", "all-sky surface shortwave downward irradiance, hourly total"),
}
START_YEAR = 2024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end else (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    man = Manifest(MAN / "raw_manifest.json")
    for region, (lat, lon, desc) in LOCATIONS.items():
        frames, shas = [], {}
        for year in range(START_YEAR, end.year + 1):
            s = datetime(year, 1, 1)
            e = min(datetime(year, 12, 31), end.replace(tzinfo=None))
            rel = f"{region}_{year}.json"
            dest = RAW / rel
            final = e < (now - timedelta(days=10)).replace(tzinfo=None)
            if not (dest.exists() and final):
                q = {"parameters": ",".join(PARAMS), "community": "RE", "longitude": lon, "latitude": lat,
                     "start": f"{s:%Y%m%d}", "end": f"{e:%Y%m%d}", "format": "JSON", "time-standard": "UTC"}
                try:
                    r = http_get(API, params=q, retries=5, backoff=15.0, timeout=300)
                except Blocked as exc:
                    print("blocked:", exc)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                man.record(rel, dest, r.url)
                print(rel, len(r.content), flush=True)
                time.sleep(5)
            else:
                man.record(rel, dest)
            shas[rel] = man.sha(rel)
            j = json.loads(dest.read_text())
            par = j["properties"]["parameter"]
            fill = j["header"]["fill_value"]
            d = pd.DataFrame({PARAMS[k][0]: pd.Series(v) for k, v in par.items()})
            d.index = pd.to_datetime(d.index, format="%Y%m%d%H", utc=True)
            frames.append(d.replace(fill, np.nan))
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC", name="time")).astype("float64")
        save_series(df, f"public_weather_nasa_{region}", {
            "freq": "1h",
            "columns": {v[0]: {"unit": v[1], "description": v[2]} for v in PARAMS.values()},
            "source": {"title": "NASA POWER Hourly API v2 (SYN1DEG/GEOS/MERRA-2 based), point data", "url": API,
                       "license": "NASA POWER data: open; cite per https://power.larc.nasa.gov/docs/referencing/",
                       "retrieved_utc": utc_now(), "raw_sha256": shas},
            "location": {"name": desc, "latitude": lat, "longitude": lon},
            "data_kind": "public", "producer": "scripts/fetch_nasa_power.py",
            "notes": "ACTUALS (reanalysis-assimilated), not archived forecasts. NASA fill value -999 mapped to NaN. Timestamp is the hour label returned by the API (UTC); the API response does not state whether it marks the start or end of the averaging hour."})
        print(f"saved {region}: {len(df)} rows, missing {float(df.isna().mean().mean()):.4f}")
    man.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
