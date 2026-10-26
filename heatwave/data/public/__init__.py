"""Loaders for the public datasets collected by ``scripts/fetch_*.py``.

Every loader returns UTC-indexed pandas DataFrames written by ``heatwave.data.schema.save_series``
(units are in the ``.meta.json`` sidecar; ``load_meta`` returns it). Sources, licences and retrieval
dates: ``data/SOURCES.md``.

Time scales
  sub-second/seconds : ``load_gpu_series`` (MIT Supercloud V100 nodes, 1 s), ``load_node_power`` (H100/B200 8-GPU nodes, 20 ms and 1 s),
                       ``load_nlr_run`` (H100 multi-node jobs, 0.2 s / 0.1 s, relative time)
  minutes : ``load_carbon("caiso")``, ``load_price("caiso", "rtm")`` (5 min)
  hours   : prices (1 h), carbon (30 min / 1 h), weather (1 h)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from heatwave.data.schema import PROCESSED, load_series

GPU_DIR = PROCESSED / "public_gpu_mit"
TRACE_DIR = PROCESSED / "public_job_traces"
NODE_DIRS = {"20ms": PROCESSED / "public_gpu_node_20ms", "1s": PROCESSED / "public_gpu_node_1s"}
NLR_DIR = PROCESSED / "public_gpu_nlr"

_REGION = {"caiso": "caiso", "np15": "caiso", "caiso_np15": "caiso", "ca": "caiso",
           "de": "de", "de-lu": "de", "de_lu": "de", "germany": "de",
           "gb": "gb", "uk": "gb", "london": "gb"}
_WEATHER_REGION = {"bayarea": "bayarea", "caiso": "bayarea", "np15": "bayarea", "ca": "bayarea",
                   "london": "london", "gb": "london", "uk": "london",
                   "frankfurt": "frankfurt", "de": "frankfurt", "de-lu": "frankfurt"}


def _region(region: str) -> str:
    try:
        return _REGION[region.lower()]
    except KeyError:
        raise ValueError(f"unknown region {region!r}; use one of caiso, de, gb") from None


def load_meta(name: str, root: Path = PROCESSED) -> dict:
    return json.loads((root / f"{name}.meta.json").read_text())


# ----------------------------------------------------------------------------- GPU telemetry
def load_gpu_jobs() -> pd.DataFrame:
    """One row per selected MIT Supercloud GPU job (id, nodes, GPUs, start/end, model label, series names)."""
    return pd.read_parquet(PROCESSED / "public_gpu_mit_jobs.parquet")


def list_gpu_series() -> list[str]:
    return sorted(p.name[: -len(".meta.json")] for p in GPU_DIR.glob("gpu_*.meta.json"))


def load_gpu_series(job_id: int, node: str | None = None) -> pd.DataFrame:
    """1 s telemetry of one job on one node: ``g<k>_<metric>`` per GPU plus ``node_*`` aggregates.

    ``node`` may be omitted for single-node jobs. Gaps are NaN.
    """
    jobs = load_gpu_jobs()
    row = jobs[jobs.job_id == int(job_id)]
    if row.empty:
        raise KeyError(f"job {job_id} not in the collected subset")
    series = list(row.iloc[0].series)
    if node is None:
        if len(series) > 1:
            raise ValueError(f"job {job_id} ran on {len(series)} nodes {list(row.iloc[0].node_ids)}; pass node=")
        name = series[0]
    else:
        name = f"gpu_{int(job_id)}_{node}"
        if name not in series:
            raise KeyError(f"{name} not found; available: {series}")
    return load_series(name, GPU_DIR)[0]


def load_gpu_meta(job_id: int, node: str | None = None) -> dict:
    jobs = load_gpu_jobs()
    series = list(jobs[jobs.job_id == int(job_id)].iloc[0].series)
    name = series[0] if node is None else f"gpu_{int(job_id)}_{node}"
    return load_meta(name, GPU_DIR)


def list_node_power_sessions() -> pd.DataFrame:
    """The 32 unique 8-GPU H100 / B200 training sessions (Elsayed et al. 2026), with hardware, task and hyperparameters."""
    return pd.read_parquet(PROCESSED / "public_gpu_node_sessions.parquet")


def load_node_power(session: str, freq: str = "1s") -> pd.DataFrame:
    """One 15-minute session of an 8-GPU node. freq='20ms': per-GPU power_w + node_power_w; freq='1s': all metrics, 1 s means."""
    return load_series(session, NODE_DIRS[freq])[0]


def list_nlr_runs() -> pd.DataFrame:
    """Runs of the NLR GenAI power dataset (H100 nodes): training (0.2 s) and sampled online inference (0.1 s). Relative time axis."""
    return pd.read_parquet(PROCESSED / "public_gpu_nlr_runs.parquet")


def load_nlr_run(run: str, freq: str | None = None) -> pd.DataFrame:
    """Total (GPU+CPU, all nodes) power of one run. The index is 1970-01-01 + seconds since job start (not a calendar time).

    freq=None keeps the native step; freq='1s' returns the mean over each second.
    """
    df = load_series(run, NLR_DIR)[0]
    return df if freq is None else df.resample(freq).mean()


def load_job_trace(name: str) -> pd.DataFrame:
    """Alibaba job traces: ``alibaba2023_pods`` (with start/end), ``alibaba2023_nodes``, ``alibaba2026_gpu_pods_sample`` (sizes, durations)."""
    return pd.read_parquet(TRACE_DIR / f"{name}.parquet")


# ----------------------------------------------------------------------------- prices
def load_price(region: str, market: str = "dam", native: bool = False) -> pd.DataFrame:
    """Electricity price. caiso: market 'dam' (1 h, USD/MWh) or 'rtm' (5 min). de: 'dam' (1 h, EUR/MWh; native=True gives the 15-min era only).

    Day-ahead prices are published the day before delivery, so 'dam' columns are usable as known-future covariates
    for real-time prices (caiso) or for the next day's load decisions.
    """
    r = _region(region)
    market = market.lower()
    if r == "caiso" and market in ("dam", "rtm"):
        return load_series(f"public_caiso_np15_lmp_{market}")[0]
    if r == "de" and market == "dam":
        return load_series("public_de_price_dam_native" if native else "public_de_price_dam_1h")[0]
    raise ValueError(f"no {market!r} price for region {region!r} (caiso: dam, rtm; de: dam)")


# ----------------------------------------------------------------------------- carbon
def load_carbon(region: str, kind: str = "actual") -> pd.DataFrame:
    """Grid carbon intensity as one column ``intensity_g_per_kwh`` (gCO2/kWh; kgCO2/MWh is numerically equal).

    gb    : 30 min; actual and forecast (the API's own forecast value; vintage undocumented).
    caiso : 5 min; actual only, computed from CAISO's published CO2 and fuel mix (see meta).
    de    : 15 min; actual only, a production-based lifecycle proxy (IPCC AR5 medians; see meta).
    """
    r = _region(region)
    kind = kind.lower()
    if r == "gb":
        col = {"actual": "intensity_actual_g_per_kwh", "forecast": "intensity_forecast_g_per_kwh"}[kind]
        return load_series("public_uk_carbon_intensity")[0][[col]].rename(columns={col: "intensity_g_per_kwh"})
    if kind != "actual":
        raise ValueError(f"{region!r} has no published intensity forecast; use load_forecast_covariates({region!r}) for day-ahead load/renewables")
    if r == "caiso":
        return load_series("public_caiso_carbon")[0][["intensity_kg_per_mwh"]].rename(columns={"intensity_kg_per_mwh": "intensity_g_per_kwh"})
    return load_series("public_de_generation")[0][["intensity_lifecycle_proxy_g_per_kwh"]].rename(
        columns={"intensity_lifecycle_proxy_g_per_kwh": "intensity_g_per_kwh"})


def load_forecast_covariates(region: str) -> pd.DataFrame:
    """Day-ahead forecasts published before delivery (known-future covariates): caiso demand, de load/solar/wind."""
    r = _region(region)
    if r == "caiso":
        return load_series("public_caiso_demand")[0][["demand_forecast_day_ahead_mw"]]
    if r == "de":
        return load_series("public_de_forecast_dayahead")[0]
    raise ValueError("gb: the carbon-intensity forecast is returned by load_carbon('gb', 'forecast')")


def load_generation_mix(region: str) -> pd.DataFrame:
    r = _region(region)
    name = {"caiso": "public_caiso_fuelmix", "gb": "public_uk_generation_mix", "de": "public_de_generation"}[r]
    return load_series(name)[0]


# ----------------------------------------------------------------------------- weather
def load_weather(region: str, kind: str = "actual", source: str | None = None, vintage: str = "short") -> pd.DataFrame:
    """Hourly weather at 'bayarea', 'london' or 'frankfurt'.

    kind='actual'  : source 'open_meteo' (ERA5-based) if collected, else 'nasa' (NASA POWER). Pass source= to force one.
    kind='forecast': Open-Meteo archived forecasts; vintage='short' (best-match, stitched short lead)
                     or 'dayahead' (previous_day1: forecast issued one day earlier).
    Raises FileNotFoundError if that file has not been collected (see data/SOURCES.md).
    """
    reg = _WEATHER_REGION[region.lower()]
    kind = kind.lower()
    if kind == "actual":
        order = [source] if source else ["open_meteo", "nasa"]
        for s in order:
            name = f"public_weather_om_actual_{reg}" if s == "open_meteo" else f"public_weather_nasa_{reg}"
            if (PROCESSED / f"{name}.parquet").exists():
                return load_series(name)[0]
        raise FileNotFoundError(f"no actual weather file for {reg} (tried {order})")
    if kind == "forecast":
        tag = {"short": "forecast", "dayahead": "dayahead"}[vintage]
        name = f"public_weather_om_{tag}_{reg}"
        if not (PROCESSED / f"{name}.parquet").exists():
            raise FileNotFoundError(f"{name} not collected (Open-Meteo quota); run scripts/fetch_open_meteo.py")
        return load_series(name)[0]
    raise ValueError("kind must be 'actual' or 'forecast'")
