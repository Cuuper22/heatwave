"""CAISO: OASIS locational marginal prices and "Today's Outlook" fuel mix / CO2 / demand history.

OASIS API spec : https://www.caiso.com/documents/oasis-interfacespecification_v5_1_3clean_fall2017release.pdf
Outlook history: https://www.caiso.com/outlook/history/<YYYYMMDD>/{fuelsource,co2,demand}.csv  (5 min, Pacific local time)
Emissions page : https://www.caiso.com/todays-outlook/emissions (units mTCO2/h and mTCO2/MWh)
CAISO terms    : https://www.caiso.com/Pages/privacy.aspx (site content is public; data is "subject to change without notice")
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

OASIS_URL = "https://oasis.caiso.com/oasisapi/SingleZip"
OUTLOOK_URL = "https://www.caiso.com/outlook/history/{day}/{name}.csv"
NODE = "TH_NP15_GEN-APND"  # NP15 generation trading hub (Northern California)
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
CHUNK_DAYS = 30
OASIS_SLEEP_S = 6.0

LMP_TYPES = {
    "LMP": ("lmp_usd_per_mwh", "total locational marginal price"),
    "MCE": ("energy_usd_per_mwh", "marginal energy component"),
    "MCC": ("congestion_usd_per_mwh", "marginal congestion component"),
    "MCL": ("loss_usd_per_mwh", "marginal loss component"),
    "MGHG": ("ghg_usd_per_mwh", "marginal greenhouse-gas component"),
}


def chunk_starts(end: datetime) -> list[tuple[datetime, datetime]]:
    out = []
    s = START
    while s < end:
        e = min(s + timedelta(days=CHUNK_DAYS), end)
        out.append((s, e))
        s = e
    return out


def oasis_params(query: str, market: str, version: str, s: datetime, e: datetime) -> dict[str, str]:
    f = "%Y%m%dT%H:%M-0000"
    return {
        "queryname": query, "market_run_id": market, "node": NODE, "version": version,
        "startdatetime": s.strftime(f), "enddatetime": e.strftime(f), "resultformat": "6",
    }


def read_oasis_zip(content: bytes) -> pd.DataFrame:
    """CSV inside the OASIS zip -> DataFrame; an XML member means an error/empty answer (returns empty frame)."""
    z = zipfile.ZipFile(io.BytesIO(content))
    names = z.namelist()
    csvs = [n for n in names if n.lower().endswith(".csv")]
    if not csvs:
        return pd.DataFrame()
    return pd.read_csv(z.open(csvs[0]))


def oasis_to_wide(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    val = "MW" if "MW" in df.columns else "VALUE"
    if "PRC" in df.columns and val not in df.columns:
        val = "PRC"
    df = df.copy()
    df["time"] = pd.to_datetime(df["INTERVALSTARTTIME_GMT"], utc=True)
    wide = df.pivot_table(index="time", columns="LMP_TYPE", values=val, aggfunc="mean")
    wide = wide.rename(columns={k: v[0] for k, v in LMP_TYPES.items()})
    return wide.sort_index()


def regularize(df: pd.DataFrame, freq: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    idx = pd.date_range(start or df.index.min(), end or df.index.max(), freq=freq, tz="UTC", name="time")
    return df[~df.index.duplicated(keep="last")].reindex(idx)


def parse_outlook_day(text: str, day: str) -> pd.DataFrame:
    """One Outlook CSV (Time = Pacific wall clock HH:MM) -> UTC-indexed frame.

    On the autumn DST change the wall-clock hour 01:xx occurs twice; the first pass is PDT.
    """
    d = pd.read_csv(io.StringIO(text))
    d = d[d["Time"].astype(str).str.match(r"^\d{1,2}:\d{2}$")].copy()
    if d.empty:
        return pd.DataFrame()
    wall = pd.to_datetime(day + " " + d["Time"].astype(str), format="%Y%m%d %H:%M")
    first = ~wall.duplicated(keep="first").to_numpy()  # True on first pass -> DST (ambiguous=True)
    idx = pd.DatetimeIndex(wall).tz_localize("America/Los_Angeles", ambiguous=first, nonexistent="shift_forward")
    d = d.drop(columns=["Time"])
    d.index = idx.tz_convert("UTC")
    d.index.name = "time"
    return d.apply(pd.to_numeric, errors="coerce")


def carbon_from_outlook(fuel: pd.DataFrame, co2: pd.DataFrame) -> pd.DataFrame:
    """Total CO2 (mTCO2/h, summed over the CAISO CO2 resources) and a computed intensity.

    supply_mw = sum of all fuel-source columns (net: battery charging is negative).
    intensity_kg_per_mwh = 1000 * co2_total_mt_per_h / supply_mw, computed here (not published by CAISO); negative while the
    published imports CO2 is negative (net export). intensity_in_state_kg_per_mwh excludes imports and is never negative.
    """
    out = pd.DataFrame(index=fuel.index.union(co2.index))
    co2c = [c for c in co2.columns]
    out = out.join(co2.rename(columns={c: "co2_" + c.lower().replace(" co2", "").replace(" ", "_") + "_mt_per_h" for c in co2c}))
    cols = [c for c in out.columns if c.startswith("co2_")]
    out["co2_total_mt_per_h"] = out[cols].sum(axis=1, min_count=len(cols))
    out["supply_mw"] = fuel.sum(axis=1, min_count=fuel.shape[1]).reindex(out.index)
    out["intensity_kg_per_mwh"] = np.where(out["supply_mw"] > 0, 1000.0 * out["co2_total_mt_per_h"] / out["supply_mw"], np.nan)
    # in-state basis: CO2 of the in-state resources over in-state generation (imports excluded, battery discharge counted).
    # The published imports CO2 is negative while California is a net exporter, which makes the total-basis value negative then.
    in_state_co2 = out[[c for c in cols if "imports" not in c]].sum(axis=1, min_count=1)
    gen_cols = [c for c in fuel.columns if "imports" not in c]
    in_state_mw = fuel[gen_cols].clip(lower=0).sum(axis=1, min_count=len(gen_cols)).reindex(out.index)
    out["in_state_generation_mw"] = in_state_mw
    out["intensity_in_state_kg_per_mwh"] = np.where(in_state_mw > 0, 1000.0 * in_state_co2 / in_state_mw, np.nan)
    return out
