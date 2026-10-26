#!/usr/bin/env python
"""Collect hourly weather from Open-Meteo for three points: actuals (archive) and archived forecasts.

Sources (https://open-meteo.com/en/docs ; data CC BY 4.0, non-commercial use on the free API tier, see
https://open-meteo.com/en/terms and https://open-meteo.com/en/license):
  archive            https://archive-api.open-meteo.com/v1/archive              ERA5/ERA5-Land based actuals
  historical_forecast https://historical-forecast-api.open-meteo.com/v1/forecast archived model forecasts (best match, stitched short-lead runs)
  previous_day1      https://previous-runs-api.open-meteo.com/v1/forecast        *_previous_day1: the forecast issued one day earlier (day-ahead vintage)

Raw   : data/raw/open_meteo/<kind>_<region>_<start>.json (gitignored); sha256 in data/processed/public_open_meteo/raw_manifest.json
Output: data/processed/public_weather_om_{actual,forecast,dayahead}_<region>.*
The free API has per-minute/hour/day request limits. This script waits between requests, backs off on 429, and
stops with exit code 3 when the daily limit is reported; re-run later (finished chunks are kept).
Wet-bulb is computed (Stull 2011), not requested.
Usage : python scripts/fetch_open_meteo.py [--kinds archive,historical_forecast,previous_day1] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from heatwave.data.public._common import PROCESSED_ROOT, RAW_ROOT, USER_AGENT, Blocked, Manifest, utc_now
from heatwave.data.public.weather import LOCATIONS, wet_bulb_stull
from heatwave.data.schema import save_series

RAW = RAW_ROOT / "open_meteo"
MAN = PROCESSED_ROOT / "public_open_meteo"
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m", "shortwave_radiation"]
OUTNAME = {
    "temperature_2m": ("temperature_2m_c", "degC", "air temperature at 2 m"),
    "relative_humidity_2m": ("relative_humidity_2m_pct", "%", "relative humidity at 2 m"),
    "dew_point_2m": ("dew_point_2m_c", "degC", "dew point at 2 m"),
    "wind_speed_10m": ("wind_speed_10m_mps", "m/s", "wind speed at 10 m (requested with wind_speed_unit=ms)"),
    "shortwave_radiation": ("shortwave_radiation_w_m2", "W/m2", "shortwave solar radiation, mean of the preceding hour"),
}
KINDS = {
    "archive": ("https://archive-api.open-meteo.com/v1/archive", "", "actual", "actual"),
    "historical_forecast": ("https://historical-forecast-api.open-meteo.com/v1/forecast", "", "forecast", "forecast"),
    "previous_day1": ("https://previous-runs-api.open-meteo.com/v1/forecast", "_previous_day1", "dayahead", "forecast"),
}
START = date(2024, 1, 1)
STEP_DAYS = 183
SLEEP_S = 15.0


class DailyLimit(RuntimeError):
    pass


def get(url: str, params: dict) -> requests.Response:
    wait = 30.0
    for _ in range(6):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=180)
        except requests.RequestException as exc:  # transient proxy/TLS hiccup
            print(f"network error, retrying: {type(exc).__name__}", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (401, 403, 407):
            raise Blocked(r.url, r.status_code)
        if r.status_code == 429:
            reason = ""
            try:
                reason = r.json().get("reason", "")
            except Exception:
                pass
            if "Daily" in reason:
                raise DailyLimit(reason)
            time.sleep(wait)
            wait *= 2
            continue
        if r.status_code >= 500:
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"giving up on {url}")


def build(kind: str, region: str, man: Manifest, end: date) -> tuple[pd.DataFrame, dict[str, str]] | None:
    url, suffix, _, _ = KINDS[kind]
    lat, lon, _ = LOCATIONS[region]
    frames, shas = [], {}
    s = START
    while s <= end:
        e = min(s + timedelta(days=STEP_DAYS - 1), end)
        rel = f"{kind}_{region}_{s:%Y%m%d}.json"
        dest = RAW / rel
        final = e < date.today() - timedelta(days=7)
        if not (dest.exists() and final):
            q = {"latitude": lat, "longitude": lon, "start_date": s.isoformat(), "end_date": e.isoformat(),
                 "hourly": ",".join(v + suffix for v in VARS), "timezone": "UTC", "wind_speed_unit": "ms"}
            r = get(url, q)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            man.record(rel, dest, r.url)
            man.save()
            print(rel, len(r.content), flush=True)
            time.sleep(SLEEP_S)
        else:
            man.record(rel, dest)
        shas[rel] = man.sha(rel)
        h = json.loads(dest.read_text())["hourly"]
        d = pd.DataFrame({OUTNAME[v][0]: h[v + suffix] for v in VARS}, index=pd.to_datetime(h["time"], utc=True))
        frames.append(d)
        s = e + timedelta(days=1)
    if not frames:
        return None
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index().astype("float64")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC", name="time"))
    df["wet_bulb_2m_c"] = wet_bulb_stull(df["temperature_2m_c"], df["relative_humidity_2m_pct"])
    return df, shas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="archive,historical_forecast,previous_day1")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=1)
    man = Manifest(MAN / "raw_manifest.json")
    status = 0
    for kind in a.kinds.split(","):
        url, _, tag, nature = KINDS[kind]
        for region, (lat, lon, desc) in LOCATIONS.items():
            try:
                out = build(kind, region, man, end)
            except DailyLimit as exc:
                print(f"STOP daily limit on {kind}/{region}: {exc}")
                return 3
            except Blocked as exc:
                print(f"BLOCKED {kind}/{region}: {exc}")
                status = 2
                continue
            if out is None:
                continue
            df, shas = out
            cols = {v[0]: {"unit": v[1], "description": v[2]} for v in OUTNAME.values()}
            cols["wet_bulb_2m_c"] = {"unit": "degC", "description": "wet-bulb temperature computed here with Stull (2011, doi:10.1175/JAMC-D-11-0143.1) from temperature_2m_c and relative_humidity_2m_pct"}
            note = {
                "actual": "ERA5-based reanalysis actuals served by the Open-Meteo archive API.",
                "forecast": "Archived forecasts from the Open-Meteo historical-forecast API: best-match model runs, stitched from short lead times. Known only up to the issue time of each run, so usable as a short-lead covariate, not as a verified day-ahead vintage.",
                "dayahead": "previous_day1 variables: the forecast issued one day before the valid time (day-ahead vintage) from the previous-runs API.",
            }[tag]
            save_series(df, f"public_weather_om_{tag}_{region}", {
                "freq": "1h", "columns": cols,
                "source": {"title": f"Open-Meteo {kind}", "url": url, "license": "CC BY 4.0, non-commercial free tier (https://open-meteo.com/en/license)",
                           "retrieved_utc": utc_now(), "raw_sha256": shas},
                "location": {"name": desc, "latitude": lat, "longitude": lon},
                "data_kind": "public", "producer": "scripts/fetch_open_meteo.py", "notes": note})
            print(f"saved {tag}/{region}: {len(df)} rows, missing {float(df.iloc[:, :5].isna().mean().mean()):.4f}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
