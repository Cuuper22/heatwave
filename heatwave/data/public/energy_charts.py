"""Energy-Charts (Fraunhofer ISE) JSON -> regular pandas frames, plus a lifecycle intensity proxy.

Emission factors: IPCC AR5 WGIII Annex III, Table A.III.2, lifecycle median gCO2eq/kWh
https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf (p. 1335).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_SOURCE = "https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf (Table A.III.2, lifecycle median)"
FACTORS = {  # gCO2eq/kWh, lifecycle median
    "coal": 820, "gas_cc": 490, "biomass_dedicated": 230, "geothermal": 38, "hydro": 24,
    "solar_pv_utility": 48, "wind_onshore": 11, "wind_offshore": 12,
}
# Energy-Charts type -> factor key. Coal PC is used for lignite, hard coal and coal-derived gas
# (the IPCC table has one coal row). Oil, waste, "Others" and storage/trade rows have no factor and are excluded.
FACTOR_MAP = {
    "Fossil brown coal / lignite": "coal", "Fossil hard coal": "coal", "Fossil coal-derived gas": "coal",
    "Fossil gas": "gas_cc", "Biomass": "biomass_dedicated", "Geothermal": "geothermal",
    "Hydro Run-of-River": "hydro", "Hydro water reservoir": "hydro", "Solar": "solar_pv_utility",
    "Wind onshore": "wind_onshore", "Wind offshore": "wind_offshore",
}
GEN_DESC = {
    "intensity_lifecycle_proxy_g_per_kwh": "production-weighted mean of IPCC AR5 lifecycle median factors over generation types that have a factor",
    "intensity_factor_coverage_pct": "share of positive net generation (excluding storage and cross-border rows) covered by a factor",
}
SKIP_SHARE = {"Load", "Residual load", "Renewable share of load", "Renewable share of generation",
              "Cross border electricity trading", "Hydro pumped storage consumption", "Hydro pumped storage"}


def _col(name: str) -> str:
    return name.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_")


def _to_series(unix: list[int], vals: list[float | None]) -> pd.Series:
    idx = pd.to_datetime(unix, unit="s", utc=True)
    return pd.Series(pd.to_numeric(pd.Series(vals, dtype="object"), errors="coerce").to_numpy(), index=idx)


def price_frame(chunks: list[dict]) -> dict:
    s = pd.concat([_to_series(c["unix_seconds"], c["price"]) for c in chunks if c.get("unix_seconds")])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    d = s.index.to_series().diff().dt.total_seconds()
    quarter = d[d == 900]
    switch = quarter.index[0] - pd.Timedelta(minutes=15) if len(quarter) else None
    hourly = s.resample("1h").mean()
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h", tz="UTC"))
    out = {"hourly": hourly.rename("price_eur_per_mwh").to_frame(), "switch_utc": str(switch)}
    if switch is not None:
        n = s[s.index >= switch]
        n = n.reindex(pd.date_range(n.index.min(), n.index.max(), freq="15min", tz="UTC"))
        out["native"], out["native_freq"] = n.rename("price_eur_per_mwh").to_frame(), "15min"
    else:
        out["native"], out["native_freq"] = out["hourly"], "1h"
    return out


def power_frame(chunks: list[dict]) -> pd.DataFrame:
    parts = []
    for c in chunks:
        if not c.get("unix_seconds"):
            continue
        cols = {}
        for pt in c["production_types"]:
            cols[pt["name"]] = _to_series(c["unix_seconds"], pt["data"])
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC", name="time"))
    gen_cols = list(FACTOR_MAP)
    eff = df[[c for c in FACTOR_MAP if c in df.columns]].clip(lower=0)
    fac = pd.Series({c: FACTORS[FACTOR_MAP[c]] for c in eff.columns})
    covered = eff.sum(axis=1, min_count=1)
    other_cols = [c for c in df.columns if c not in SKIP_SHARE and c not in FACTOR_MAP]
    total = covered + df[other_cols].clip(lower=0).sum(axis=1, min_count=1).fillna(0)
    inten = (eff * fac).sum(axis=1, min_count=1) / covered
    out = df.rename(columns={c: _col(c) + "_mw" for c in df.columns}).copy()
    out["intensity_lifecycle_proxy_g_per_kwh"] = inten.where(covered > 0)
    out["intensity_factor_coverage_pct"] = 100.0 * covered / total.where(total > 0)
    # renaming of share rows keeps their unit: they are percent, but the _mw suffix would mislead
    out = out.rename(columns={"renewable_share_of_load_mw": "renewable_share_of_load_pct",
                              "renewable_share_of_generation_mw": "renewable_share_of_generation_pct"})
    return out.astype("float64")


def forecast_frame(chunks_by_key: dict[str, list[dict]]) -> pd.DataFrame:
    cols = {}
    for key, chunks in chunks_by_key.items():
        parts = [_to_series(c["unix_seconds"], c["forecast_values"]) for c in chunks if c.get("unix_seconds")]
        s = pd.concat(parts)
        cols[key.replace("fc_", "") + "_forecast_mw"] = s[~s.index.duplicated(keep="last")].sort_index()
    df = pd.DataFrame(cols)
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC", name="time")).astype("float64")
