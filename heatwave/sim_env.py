"""Drive the gpu_stack e003 simulator with PUBLIC weather, price and carbon series.

``public_environment(start_utc, horizon_s)`` returns, for the simulator's two
sites, ``gpu_stack.research.heatwave_sim.SiteEnvironment`` objects plus the
forecast series a controller or forecaster may legitimately use as
known-future covariates:

  site-a: Bay Area weather (Open-Meteo), CAISO NP15 day-ahead LMP, CAISO carbon
          intensity (in-state basis)
  site-b: Frankfurt weather (Open-Meteo), DE-LU day-ahead price (EUR converted
          to USD at one fixed rate passed by the caller), DE carbon intensity

Simulated plant, public inputs: results that use this are "mixed" (simulated
cluster, public exogenous series); label them so.

What is known in advance, and what is not:
  * outdoor temperature: the simulator's actual series is the Open-Meteo
    archive (ERA5-based); the forecast series is Open-Meteo's archived
    day-ahead forecast (``previous_day1``), a real forecast vintage.
  * price: day-ahead prices are published the day before delivery, so the
    sites pay DA prices and the DA series is a legitimate known-future input.
  * carbon: no day-ahead carbon forecast exists for CAISO or DE in this
    repo's data. ``carbon_forecast_g_per_kwh`` is a seasonal-naive day-ahead
    forecast (the value 24 h earlier), which only uses the past.
    The simulator's own ``known_future["carbon_g_per_kwh"]`` and its
    ``cov_*_carbon_g_per_kwh`` exports pass the ACTUAL series through, which
    would leak the future; use ``carbon_forecast_g_per_kwh`` instead.
Gaps (NaN) are filled by time interpolation, then edge fill; the share filled
is returned per series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from heatwave.data import public as P

SITES = {"site-a": ("bayarea", "caiso"), "site-b": ("frankfurt", "de")}


@dataclass(frozen=True)
class PublicEnvironment:
    environment: dict            # site id -> SiteEnvironment
    hours_utc: pd.DatetimeIndex  # timestamp of each hourly value
    forecasts: dict              # site id -> {"outdoor_forecast_c", "price_usd_per_kwh", "carbon_forecast_g_per_kwh"}
    filled_share: dict           # site id -> {series: share of hours filled}
    sources: dict


def _hourly(s: pd.Series, idx: pd.DatetimeIndex) -> tuple[np.ndarray, float]:
    h = s.resample("1h").mean().reindex(idx)
    filled = float(h.isna().mean())
    h = h.interpolate(method="time", limit_direction="both")
    return h.to_numpy(dtype=float), filled


def public_environment(start_utc: str, horizon_s: float, usd_per_eur: float) -> PublicEnvironment:
    from gpu_stack.research.heatwave_sim import SiteEnvironment

    n = int(round(horizon_s / 3600.0)) + 1
    start = pd.Timestamp(start_utc)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    idx = pd.date_range(start, periods=n, freq="1h")
    prev = pd.date_range(start - pd.Timedelta(hours=24), periods=n, freq="1h")
    env, fc, filled = {}, {}, {}
    for sid, (wreg, greg) in SITES.items():
        wa = P.load_weather(wreg, "actual")["temperature_2m_c"]
        wf = P.load_weather(wreg, "forecast", vintage="dayahead")["temperature_2m_c"]
        if greg == "caiso":
            price = P.load_price("caiso", "dam")["lmp_usd_per_mwh"] / 1000.0
        else:
            price = P.load_price("de", "dam")["price_eur_per_mwh"] * float(usd_per_eur) / 1000.0
        carbon = P.load_carbon(greg)["intensity_g_per_kwh"]
        t_act, f1 = _hourly(wa, idx)
        t_fc, f2 = _hourly(wf, idx)
        pr, f3 = _hourly(price, idx)
        ca, f4 = _hourly(carbon, idx)
        ca_prev, f5 = _hourly(carbon, prev)
        env[sid] = SiteEnvironment(outdoor_c=t_act, price_usd_per_kwh=pr, carbon_g_per_kwh=ca,
                                   outdoor_forecast_c=t_fc)
        fc[sid] = {"outdoor_forecast_c": t_fc, "price_usd_per_kwh": pr, "carbon_forecast_g_per_kwh": ca_prev}
        filled[sid] = {"outdoor_c": f1, "outdoor_forecast_c": f2, "price": f3, "carbon": f4, "carbon_forecast": f5}
    sources = {
        "weather_actual": "Open-Meteo archive (ERA5-based), data/SOURCES.md",
        "weather_forecast": "Open-Meteo previous-runs API, previous_day1 (day-ahead vintage)",
        "price": "CAISO NP15 day-ahead LMP; DE-LU day-ahead price (Energy-Charts) x usd_per_eur",
        "carbon": "CAISO in-state carbon intensity; DE carbon intensity (see data/SOURCES.md)",
        "carbon_forecast": "seasonal-naive day-ahead: the actual value 24 h earlier",
        "usd_per_eur": float(usd_per_eur),
    }
    return PublicEnvironment(env, idx, fc, filled, sources)
