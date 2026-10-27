"""The e003 Heatwave simulator on the PUBLIC environment, for fine-tuning and its held-out windows.

Plant: gpu_stack branch ``heatwave``, ``experiments/e003-heatwave/sim-scenario-v3.json`` (SIMULATED; the job schedule
follows the run seed). Environment: Open-Meteo weather (actual for the plant, the archived day-ahead forecast as
covariate), CAISO NP15 and DE-LU day-ahead prices, CAISO and DE carbon intensity, through
``heatwave.sim_env.public_environment`` (so results that use this are "mixed": simulated plant, public exogenous series).
The simulator's own ``cov_*_carbon_g_per_kwh`` and ``cov_*_price_usd_per_kwh`` exports are NOT used: they carry the
actual series, which leaks the future. Covariates come from ``SimCovariates`` below.

Run seed -> simulated week (fixed splits of the project: train seeds 0-49, validation 50-59, TEST 100-109):
  * TEST 100-109: scenario ``start_utc`` + 7 days x (seed - 100)  (2026-07-06 onward; the rule of the covariate ablation)
  * validation 50-59: 2026-01-05 + 7 days x (seed - 50)            (all before 2026-04-01)
  * train 0-49: 2024-02-05 + 14 days x seed                        (all before 2026-01-01; Open-Meteo day-ahead archive
    is complete from 2024-02-01)
So train and validation weeks lie before the public-series test period (2026-05-01 onward) and never overlap a test week.

Covariate groups (same names and channels as the covariate ablation of this repo, so that a checkpoint can be scored with
either): J job schedule (planned GPUs running, planned checkpoint GPUs), P training-step phase, W weather forecast
(Open-Meteo previous_day1), C day-ahead price and seasonal-naive carbon forecast. Leakage rules, each tested in
tests/test_finetune.py:
  * J, P: the plan uses jobs with ``submit_s <= origin`` only;
  * W: day-ahead forecast, usable from 24 h before its valid time; never the actual outdoor temperature;
  * C price: usable from 13:00 local market time on the day before delivery (assumed publication time);
    C carbon: the actual of 24 h earlier, usable 23 h before its hour;
  * every group is NaN beyond the scenario's known-future horizon (24 h) after the origin.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

SCENARIO_PATH = os.environ.get(
    "HEATWAVE_SIM_SCENARIO", "/home/claude/gpu_stack-/experiments/e003-heatwave/sim-scenario-v3.json")
GROUPS = ("J", "P", "W", "C")
GROUP_CHANNELS = {
    "J": ("planned_gpus_running", "scheduled_checkpoint_gpus"),
    "P": ("step_phase_compute_fraction",),
    "W": ("outdoor_forecast_c",),
    "C": ("price_usd_per_kwh", "carbon_forecast_g_per_kwh"),
}
SITE_MARKET_TZ = {"site-a": "America/Los_Angeles", "site-b": "Europe/Berlin"}
DAY_AHEAD_PUBLISH_LOCAL_HOUR = 13
WEATHER_FORECAST_LEAD = pd.Timedelta(hours=24)
CARBON_FORECAST_AVAILABLE_LEAD = pd.Timedelta(hours=23)

TRAIN_SEEDS = tuple(range(0, 50))
VAL_SEEDS = tuple(range(50, 60))
TEST_SEEDS = tuple(range(100, 110))
TRAIN_FIRST_WEEK = pd.Timestamp("2024-02-05", tz="UTC")
VAL_FIRST_WEEK = pd.Timestamp("2026-01-05", tz="UTC")
SPLIT_END_TRAIN = pd.Timestamp("2026-01-01", tz="UTC")     # every train week ends before this
SPLIT_END_VAL = pd.Timestamp("2026-04-01", tz="UTC")       # every validation week ends before this
TEST_FROM_UTC = pd.Timestamp("2026-05-01", tz="UTC")       # public-series test period start (fixed split)


def split_of_seed(seed: int) -> str:
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in VAL_SEEDS:
        return "val"
    if seed in TEST_SEEDS:
        return "test"
    raise ValueError(f"seed {seed} is outside the split ranges 0-49, 50-59, 100-109")


def load_scenario(path: str | Path = SCENARIO_PATH) -> dict:
    return json.loads(Path(path).read_text())


def seed_start_utc(seed: int, scenario: Optional[Mapping] = None) -> pd.Timestamp:
    sp = split_of_seed(seed)
    if sp == "test":
        sc = scenario or load_scenario()
        t = pd.Timestamp(sc["start_utc"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t + pd.Timedelta(days=7 * (seed - 100))
    if sp == "val":
        return VAL_FIRST_WEEK + pd.Timedelta(days=7 * (seed - 50))
    return TRAIN_FIRST_WEEK + pd.Timedelta(days=14 * seed)


def usd_per_eur(path: Optional[str | Path] = None) -> tuple[float, str]:
    p = Path(path) if path else Path(SCENARIO_PATH).parent / "results" / "cost-v1.json"
    return float(json.loads(p.read_text())["result"]["fx"]["usd_per_eur_mean_over_ecb_business_days"]), str(p)


_ENV_CACHE: dict = {}


def public_env(seed: int, scenario: Mapping, fx: float):
    from heatwave.sim_env import public_environment

    start = seed_start_utc(seed, scenario)
    key = (str(start), float(scenario["horizon_s"]), float(fx))
    if key not in _ENV_CACHE:
        if len(_ENV_CACHE) > 4:
            _ENV_CACHE.clear()
        _ENV_CACHE[key] = public_environment(start.isoformat(), float(scenario["horizon_s"]), float(fx))
    return _ENV_CACHE[key]


# --------------------------------------------------------------------------------------------- simulator runs
def sim_cache_file(cache_dir: str | Path, seed: int) -> Path:
    return Path(cache_dir) / f"ft_sim_v3_{seed}.npz"


def run_sim(seed: int, cache_dir: str | Path, fx: Optional[float] = None, scenario_path: str | Path = SCENARIO_PATH,
            overwrite: bool = False) -> Path:
    """Run one simulated week (SIMULATED plant, PUBLIC environment) and cache the series the windows are cut from."""
    out = sim_cache_file(cache_dir, seed)
    if out.exists() and not overwrite:
        return out
    from gpu_stack.research import heatwave_sim as hs

    sc = hs.load_scenario(scenario_path)
    sc["outputs"] = {"gpu_windows": []}
    fx = usd_per_eur()[0] if fx is None else fx
    pe = public_env(seed, sc, fx)
    t0 = time.time()
    a = hs.run_scenario(sc, seed=seed, environment=pe.environment).to_arrays()
    meta = a["meta"]
    info = {
        "scenario_id": sc["scenario_id"], "seed": int(seed), "split": split_of_seed(seed),
        "start_utc": str(seed_start_utc(seed, sc)), "usd_per_eur": fx, "filled_share": pe.filled_share,
        "sources": pe.sources, "data_kind": "mixed (simulated plant, public environment)",
    }
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        rack_power_w_1s=a["rack_power_w"].astype(np.float32),
        rack_temp_c_1min=a["rack_temp_c_1min"].astype(np.float32),
        site_mean_power_w_1h=a["site_mean_power_w_1h"].astype(np.float32),
        rack_ids=np.array(meta["rack_ids"]), site_ids=np.array(meta["site_ids"]),
        rack_site_ids=np.array(meta["rack_site_ids"]),
        info=np.array(json.dumps(info, default=str)), wall_s=np.array(time.time() - t0),
    )
    os.replace(tmp, out)
    return out


def load_sim(seed: int, cache_dir: str | Path) -> dict:
    z = np.load(sim_cache_file(cache_dir, seed))
    d = {k: z[k] for k in z.files}
    d["info"] = json.loads(str(d["info"]))
    return d


#: quantities cut from a simulator run: key in the cache, covariate level, step in seconds, units
SIM_KINDS = {
    "rack_power_1s": dict(key="rack_power_w_1s", level="rack", step_s=1.0),
    "rack_temp_1min": dict(key="rack_temp_c_1min", level="rack", step_s=60.0),
    "site_power_1h": dict(key="site_mean_power_w_1h", level="site", step_s=3600.0),
}


# --------------------------------------------------------------------------------------------- covariates
def publication_instants(grid: pd.DatetimeIndex, *, tz: str, publish_local_hour: int = DAY_AHEAD_PUBLISH_LOCAL_HOUR) -> np.ndarray:
    """For each grid step, the UTC instant (ns) at which the day-ahead product for its local day is published."""
    local_day = grid.tz_convert(tz).normalize()
    days = local_day.unique()
    pub = pd.DatetimeIndex(
        [(d.tz_localize(None) - pd.Timedelta("1D") + pd.Timedelta(f"{publish_local_hour}h")).tz_localize(
            tz, ambiguous=True, nonexistent="shift_forward") for d in days]).tz_convert("UTC")
    day_to_pub = dict(zip(days, pub.asi8))
    return np.array([day_to_pub[d] for d in local_day], dtype=np.int64)


class SimCovariates:
    """Known-future covariates of one simulator run seed, built per forecast origin (SIMULATED plant, PUBLIC exogenous series)."""

    def __init__(self, seed: int, scenario: Mapping, fx: float) -> None:
        from gpu_stack.research import heatwave_sim as hs

        self._hs = hs
        self.seed = int(seed)
        self.scenario = copy.deepcopy(dict(scenario))
        self.env = public_env(seed, self.scenario, fx)
        self.cfg = hs.build_config(self.scenario, seed=self.seed, environment=self.env.environment)
        self.known_future_horizon_s = float(self.cfg.known_future_horizon_s)
        self.rack_ids = [r.rack_id for lay in self.cfg.layouts for r in lay.racks]
        self.rack_site = [lay.site.site_id for lay in self.cfg.layouts for _ in lay.racks]
        self.site_ids = [lay.site.site_id for lay in self.cfg.layouts]
        self._jobs = sorted(self.cfg.jobs, key=lambda j: j.submit_s)
        self._submit = np.array([j.submit_s for j in self._jobs])
        self._plans: dict[int, tuple] = {}
        hours = pd.DatetimeIndex(self.env.hours_utc)
        self.start_ns = int(hours.asi8[0])
        self.n_nodes = len(hours)
        self._ext: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for sid in self.site_ids:
            f = self.env.forecasts[sid]
            self._ext[sid] = {
                "outdoor_forecast_c": (np.asarray(f["outdoor_forecast_c"], float), hours.asi8 - int(WEATHER_FORECAST_LEAD.value)),
                "price_usd_per_kwh": (np.asarray(f["price_usd_per_kwh"], float),
                                      publication_instants(hours, tz=SITE_MARKET_TZ[sid])),
                "carbon_forecast_g_per_kwh": (np.asarray(f["carbon_forecast_g_per_kwh"], float),
                                              hours.asi8 - int(CARBON_FORECAST_AVAILABLE_LEAD.value)),
            }

    def n_known_jobs(self, origin_s: float) -> int:
        return int(np.searchsorted(self._submit, origin_s, side="right"))

    def _plan(self, n_jobs: int):
        if n_jobs not in self._plans:
            cfg = replace(self.cfg, jobs=tuple(self._jobs[:n_jobs]))
            self._plans[n_jobs] = (cfg, self._hs.plan_schedule(cfg))
        return self._plans[n_jobs]

    def external(self, site: str, name: str, resolution_s: float, t0_s: float, t1_s: float, origin_s: float) -> np.ndarray:
        """Hourly forecast series ``name`` of ``site`` over bins [t0_s, t1_s) as known at ``origin_s``; NaN where not yet usable."""
        values, avail = self._ext[site][name]
        origin_ns = self.start_ns + int(round(origin_s * 1e9))
        nbins = int(round((t1_s - t0_s) / resolution_s))
        edges = t0_s + np.arange(nbins + 1) * resolution_s
        sub = max(1, int(round(resolution_s / 60.0))) if resolution_s >= 60 else 1
        mids = edges[:-1, None] + (np.arange(sub)[None, :] + 0.5) * resolution_s / sub
        node_s = np.arange(self.n_nodes) * 3600.0
        node_ns = self.start_ns + (node_s * 1e9).astype(np.int64)
        usable = np.minimum(avail, node_ns) <= origin_ns
        if name == "outdoor_forecast_c":
            out = np.interp(mids, node_s, values).mean(axis=1)
            lo = np.clip(np.floor(mids / 3600.0).astype(int), 0, self.n_nodes - 1)
            hi = np.clip(lo + 1, 0, self.n_nodes - 1)
            ok = (usable[lo] & usable[hi]).all(axis=1)
        else:
            idx = np.clip((mids // 3600.0).astype(int), 0, self.n_nodes - 1)
            out = values[idx].mean(axis=1)
            ok = usable[idx].all(axis=1)
        out = out.copy()
        out[~ok] = np.nan
        return out

    def window(self, level: str, resolution_s: float, origin_idx: int, context: int, horizon: int,
               *, full_knowledge: bool = False) -> dict[str, np.ndarray]:
        """Covariates for bins [origin_idx - context, origin_idx + horizon): {group: (C_g, units, context + horizon)}.

        Horizon steps that end more than ``known_future_horizon_s`` after the origin are NaN in every group.
        ``full_knowledge=True`` plans with every job of the scenario, including jobs submitted after the origin; it exists
        only so that a test can show the difference."""
        origin_s = origin_idx * resolution_s
        t0 = (origin_idx - context) * resolution_s
        t1 = (origin_idx + horizon) * resolution_s
        if t0 < 0 or t1 > self.cfg.horizon_s + 1e-9:
            raise ValueError("window outside the simulated span")
        n = len(self._jobs) if full_knowledge else self.n_known_jobs(origin_s)
        cfg, plans = self._plan(n)
        cov = self._hs.planned_covariates(cfg, plans, resolution_s, level, t0, t1)
        units = self.rack_ids if level == "rack" else self.site_ids
        unit_site = self.rack_site if level == "rack" else self.site_ids
        cache: dict[tuple[str, str], np.ndarray] = {}

        def ext(site: str, name: str) -> np.ndarray:
            if (site, name) not in cache:
                cache[(site, name)] = self.external(site, name, resolution_s, t0, t1, origin_s)
            return cache[(site, name)]

        steps_end_s = (np.arange(context + horizon) - context + 1) * resolution_s   # end of each bin relative to the origin
        beyond = steps_end_s > self.known_future_horizon_s + 1e-9
        out = {}
        for g, keys in (("J", GROUP_CHANNELS["J"]), ("P", GROUP_CHANNELS["P"]), ("W", GROUP_CHANNELS["W"]), ("C", GROUP_CHANNELS["C"])):
            if g in ("J", "P"):
                arr = np.stack([cov[k].T for k in keys], axis=0)
            else:
                arr = np.stack([np.stack([ext(unit_site[u], k) for u in range(len(units))], axis=0) for k in keys], axis=0)
            arr = np.array(arr, dtype=np.float64)
            arr[..., beyond] = np.nan
            out[g] = arr
        return out
