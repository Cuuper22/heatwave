"""Evaluation windows for the multi-scale forecasting study and the hierarchy study.

Every builder returns plain numpy windows (context, future, optional known-future covariates) cut
from the sources in data/SOURCES.md and from cached simulator runs. Splits follow the lead's fixed rules
(README of the task addendum):

* simulator: train seeds 0-49, validation seeds 50-59, TEST seeds 100-109
* Elsayed et al. 2026 node sessions: TEST = sessions whose index in ``list_node_power_sessions()``
  sorted by session name is divisible by 3
* NLR runs: TEST = runs with ``repeat`` == 4 (a (model, nodes) group without repeat 4 uses its last repeat)
* MIT Supercloud V100 jobs: TEST = job id modulo 4 == 0
* laptop: run1 = train / calibration, run2 = TEST
* grid and carbon series: TEST = origins from 2026-05-01 00:00 UTC on

Origins are chosen by a deterministic rule (equally spaced, no randomness) so a reader can recompute
which windows were used; ``Windows.meta`` records the number of admissible origins and the number used.

Data kinds: ``measured`` (laptop), ``public``, ``simulated`` (gpu_stack heatwave simulator).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from heatwave.data import public as P
from heatwave.data.schema import PROCESSED, load_series

from .hier_coherence import Hierarchy, build_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "results" / "cache" / "forecast"

# --------------------------------------------------------------------------- protocol constants
# Fixed per scale, chosen before any evaluation and never tuned on test data.
CONTEXT = {"seconds": 256, "minutes": 96, "hours": 96}   # steps
HORIZON = {"seconds": 300, "minutes": 60, "hours": 48}   # steps (1 s, 1 min or 15 s, 1 h)
SEASON = {"seconds": 1, "minutes": 1, "hours": 24}       # steps; used by MASE scale and the classical baselines
# Horizon buckets in steps of each scale's native unit: (label, first step, last step), 1-based.
BUCKETS = {
    "seconds": (("1-10s", 1, 10), ("11-60s", 11, 60), ("61-180s", 61, 180), ("181-300s", 181, 300)),
    "minutes": (("1-5min", 1, 5), ("6-15min", 6, 15), ("16-30min", 16, 30), ("31-60min", 31, 60)),
    "hours": (("1-6h", 1, 6), ("7-24h", 7, 24), ("25-48h", 25, 48)),
}

TEST_FROM_UTC = pd.Timestamp("2026-05-01", tz="UTC")
SIM_SCENARIO = os.environ.get("HEATWAVE_SIM_SCENARIO", "/home/claude/gpu_stack-/experiments/e003-heatwave/sim-scenario-v3.json")
SIM_TAG = Path(SIM_SCENARIO).stem.replace("sim-scenario-", "sim_")  # e.g. sim_v3: cache files are per scenario
SIM_TEST_SEEDS = tuple(range(100, 110))
SIM_VAL_SEEDS = tuple(range(50, 54))     # subset of the validation seeds 50-59, used for MinT covariance only
SIM_START_UTC = pd.Timestamp("2026-07-06", tz="UTC")  # scenario start_utc of sim-scenario-v2/v3.json
# Windows of per-GPU power (start second, end second), fixed a priori. The first four (30, 70, 110, 150 h into the week) are the
# evaluation windows of the TEST seeds; the last four (50, 90, 130, 160 h) are calibration windows: the MinT covariance is
# estimated from the validation seeds in these windows only, so it never sees the times or the seeds of the test windows.
# Note: the scenario's job schedule is generated with a fixed seed (jobs.generate.seed = 7), so every simulator seed has
# the same schedule and differs in noise and event realisation only.
SIM_GPU_WINDOWS_S = ((108000, 109800), (252000, 253800), (396000, 397800), (540000, 541800),
                     (180000, 181800), (324000, 325800), (468000, 469800), (576000, 577800))
SIM_EVAL_WINDOWS = (0, 1, 2, 3)
SIM_CALIB_WINDOWS = (4, 5, 6, 7)


# --------------------------------------------------------------------------- containers
@dataclass
class Series:
    id: str
    values: np.ndarray                      # (T,)
    step_s: float
    t0: Optional[pd.Timestamp] = None
    group: str = ""
    test_from: int = 0                      # first admissible origin index (0-based index of the first forecast step)
    cov: Optional[np.ndarray] = None        # (C, T) known-future covariates on the same grid
    cov_names: tuple[str, ...] = ()


@dataclass
class Windows:
    name: str
    scale: str
    data_kind: str
    step_s: float
    season: int
    context: np.ndarray                     # (N, T)
    future: np.ndarray                      # (N, H)
    series_id: np.ndarray                   # (N,) str
    group: np.ndarray                       # (N,) str, the independent unit for cluster bootstrap
    origin_idx: np.ndarray                  # (N,) index of the first forecast step in its series
    origin_time: np.ndarray                 # (N,) ISO string or ""
    known_cov: Optional[np.ndarray] = None  # (N, C, T + H)
    cov_names: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.context.shape[0]

    def subset(self, idx: Sequence[int]) -> "Windows":
        idx = np.asarray(idx, dtype=int)
        return replace(
            self, context=self.context[idx], future=self.future[idx], series_id=self.series_id[idx],
            group=self.group[idx], origin_idx=self.origin_idx[idx], origin_time=self.origin_time[idx],
            known_cov=None if self.known_cov is None else self.known_cov[idx],
        )


@dataclass
class HierWindows:
    name: str
    data_kind: str
    step_s: float
    season: int
    hierarchy: Hierarchy
    context: np.ndarray                     # (N, n, T)
    future: np.ndarray                      # (N, n, H)
    window_id: np.ndarray                   # (N,) str
    group: np.ndarray                       # (N,) str
    split: str = "test"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.context.shape[0]


# --------------------------------------------------------------------------- window cutting
def origin_positions(T: int, context: int, horizon: int, stride: int, first: int = 0, max_n: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Admissible origins context <= o <= T - horizon (and o >= ``first``), every ``stride`` steps; thinned to
    ``max_n`` equally spaced positions. Returns (positions, number admissible before thinning)."""
    lo = max(context, int(first))
    hi = T - horizon
    if hi < lo:
        return np.empty(0, dtype=int), 0
    pos = np.arange(lo, hi + 1, stride, dtype=int)
    n_adm = len(pos)
    if max_n is not None and n_adm > max_n:
        pos = pos[np.unique(np.round(np.linspace(0, n_adm - 1, max_n)).astype(int))]
    return pos, n_adm


def make_windows(
    name: str, scale: str, data_kind: str, series: Sequence[Series], *, stride: int,
    max_per_series: Optional[int] = None, context: Optional[int] = None, horizon: Optional[int] = None,
    max_context_nan_frac: float = 0.1, meta: Optional[dict] = None,
) -> Windows:
    T = context or CONTEXT[scale]
    H = horizon or HORIZON[scale]
    ctxs, futs, sid, grp, oi, ot, covs = [], [], [], [], [], [], []
    n_adm_total = 0
    n_dropped = 0
    cov_names: tuple[str, ...] = ()
    step_s = series[0].step_s if series else 1.0
    for s in series:
        pos, n_adm = origin_positions(len(s.values), T, H, stride, s.test_from, max_per_series)
        n_adm_total += n_adm
        for o in pos:
            c = s.values[o - T:o]
            f = s.values[o:o + H]
            if not np.all(np.isfinite(f)) or np.mean(~np.isfinite(c)) > max_context_nan_frac:
                n_dropped += 1
                continue
            ctxs.append(c)
            futs.append(f)
            sid.append(s.id)
            grp.append(s.group or s.id)
            oi.append(o)
            ot.append("" if s.t0 is None else (s.t0 + pd.Timedelta(float(o * s.step_s), unit="s")).isoformat())
            if s.cov is not None:
                covs.append(s.cov[:, o - T:o + H])
                cov_names = s.cov_names
    m = dict(meta or {})
    m.update({
        "n_series": len(series), "n_admissible_origins": n_adm_total, "n_windows": len(ctxs), "n_dropped_nonfinite": n_dropped,
        "stride_steps": stride, "max_origins_per_series": max_per_series, "context_steps": T, "horizon_steps": H,
    })
    return Windows(
        name=name, scale=scale, data_kind=data_kind, step_s=step_s, season=SEASON[scale],
        context=np.asarray(ctxs, dtype=np.float64).reshape(len(ctxs), T),
        future=np.asarray(futs, dtype=np.float64).reshape(len(futs), H),
        series_id=np.asarray(sid, dtype=object), group=np.asarray(grp, dtype=object),
        origin_idx=np.asarray(oi, dtype=int), origin_time=np.asarray(ot, dtype=object),
        known_cov=np.asarray(covs, dtype=np.float64) if covs else None, cov_names=cov_names, meta=m,
    )


def subsample_index(n: int, k: int) -> np.ndarray:
    """Equally spaced, deterministic indices of ``k`` out of ``n`` items (all of them if k >= n)."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


def subsample_windows(w: Windows, n: int) -> Windows:
    """Equally spaced subset of ``n`` windows (deterministic). Records the subsampling in ``meta``."""
    if w.n <= n:
        return w
    idx = subsample_index(w.n, n)
    out = w.subset(idx)
    out.meta = {**w.meta, "subsample_of": w.n, "subsample_n": int(len(idx))}
    return out


# --------------------------------------------------------------------------- measured: laptop
def laptop_series(run: str, quantity: str, with_covariates: bool = True) -> list[Series]:
    """Measured laptop RTX 3060 Laptop GPU. quantity 'power' (1 s, energy-counter power: resolves checkpoint dips)
    or 'temp' (15 s means of GPU temperature). Known-future covariates: training_on, ckpt_scheduled (from the
    pre-written load schedule)."""
    d, _ = load_series(f"laptop_{run}_1s")
    d = d.resample("1s").mean()
    step = "1s" if quantity == "power" else "15s"
    step_s = 1.0 if quantity == "power" else 15.0
    col = "power_energy_w" if quantity == "power" else "temp_c"
    y = d[col] if quantity == "power" else d[col].resample(step).mean()
    cov = None
    names: tuple[str, ...] = ()
    if with_covariates:
        c1 = d["training_on"].fillna(0.0)
        c2 = d["ckpt_scheduled"].fillna(0.0)
        if quantity != "power":
            c1 = c1.resample(step).mean()
            c2 = c2.resample(step).max()
        cov = np.vstack([c1.reindex(y.index).fillna(0.0).to_numpy(), c2.reindex(y.index).fillna(0.0).to_numpy()])
        names = ("training_on", "ckpt_scheduled")
    return [Series(id=f"laptop_{run}_{quantity}", values=y.to_numpy(dtype=np.float64), step_s=step_s, t0=y.index[0],
                   group=f"laptop_{run}", cov=cov, cov_names=names)]


# --------------------------------------------------------------------------- public GPU telemetry
def node_sessions(split: str) -> list[str]:
    s = P.list_node_power_sessions().sort_values("session").reset_index(drop=True)
    keep = [x for i, x in enumerate(s.session) if (i % 3 == 0) == (split == "test")]
    return keep


def elsayed_gpu_series(split: str = "test", quantity: str = "power") -> list[Series]:
    """Per-GPU series of 8-GPU H100/B200 nodes (Elsayed et al. 2026), 1 s means."""
    col = {"power": "power_w", "temp": "temp_c"}[quantity]
    out = []
    for sess in node_sessions(split):
        d = P.load_node_power(sess, "1s")
        for k in range(8):
            out.append(Series(id=f"{sess}/g{k}", values=d[f"g{k}_{col}"].to_numpy(dtype=np.float64), step_s=1.0,
                              t0=d.index[0], group=sess))
    return out


def nlr_run_names(split: str) -> list[str]:
    r = P.list_nlr_runs()
    r = r[r.kind == "training"].copy()
    test = []
    for _, g in r.groupby(["model", "nodes"]):
        reps = g.repeat.to_numpy()
        want = 4.0 if (reps == 4.0).any() else reps.max()
        test += g[g.repeat == want].run.tolist()
    return sorted(test) if split == "test" else sorted(set(r.run) - set(test))


def nlr_series(split: str = "test") -> list[Series]:
    """NLR 2026 H100 multi-node training jobs: total power, mean per second. The time axis is relative (t0 = None)."""
    out = []
    for run in nlr_run_names(split):
        d = P.load_nlr_run(run, freq="1s")
        out.append(Series(id=run, values=d["total_power_w"].to_numpy(dtype=np.float64), step_s=1.0, t0=None, group=run))
    return out


def mit_job_ids(split: str) -> list[int]:
    j = P.load_gpu_jobs()
    ids = j.job_id.astype(np.int64)
    return sorted(int(x) for x in ids[(ids % 4 == 0) == (split == "test")])


def mit_series(split: str = "test", quantity: str = "power") -> list[Series]:
    """MIT Supercloud V100 nodes (local copy, CC BY-NC-ND 4.0, never committed). power: node power at 1 s.
    temp: node maximum GPU temperature, 1-minute means."""
    jobs = P.load_gpu_jobs().set_index("job_id")
    out = []
    for jid in mit_job_ids(split):
        row = jobs.loc[jid]
        for name in list(row.series):
            d = load_series(name, P.GPU_DIR)[0]
            if quantity == "power":
                y, step_s = d["node_power_w"], 1.0
            else:
                y, step_s = d["node_temp_c_max"].resample("1min").mean(), 60.0
            out.append(Series(id=name, values=y.to_numpy(dtype=np.float64), step_s=step_s, t0=y.index[0], group=str(jid)))
    return out


# --------------------------------------------------------------------------- grid and carbon (hours)
def _hourly(s: pd.Series, per_hour: int, min_frac: float = 0.8) -> pd.Series:
    r = s.resample("1h")
    out = r.mean()
    cnt = r.count()
    return out.where(cnt >= min_frac * per_hour)


def _weather_cov(region: str, index: pd.DatetimeIndex) -> tuple[Optional[np.ndarray], tuple[str, ...]]:
    """Day-ahead archived Open-Meteo forecasts as known-future covariates (None if not collected)."""
    try:
        w = P.load_weather(region, "forecast", vintage="dayahead")
    except Exception:
        return None, ()
    cols = ["temperature_2m_c", "wet_bulb_2m_c"]
    x = w[cols].reindex(index).ffill().bfill()
    return x.to_numpy(dtype=np.float64).T, tuple(f"weather_dayahead_{c}" for c in cols)


def caiso_price_series(with_covariates: bool = True) -> list[Series]:
    """CAISO NP15 real-time LMP, 5-minute prices averaged to hourly. Optional known-future covariates: the
    day-ahead LMP and the day-ahead weather forecast (Bay Area). Not used in the T04a runs (covariate ablation is T04b)."""
    rt = _hourly(P.load_price("caiso", "rtm")["lmp_usd_per_mwh"], 12)
    cov, names = None, ()
    if with_covariates:
        dam = P.load_price("caiso", "dam")["lmp_usd_per_mwh"].reindex(rt.index).ffill().bfill().to_numpy()[None, :]
        wx, wn = _weather_cov("bayarea", rt.index)
        cov = dam if wx is None else np.vstack([dam, wx])
        names = ("caiso_dam_lmp_usd_per_mwh",) + wn
    tf = int(rt.index.searchsorted(TEST_FROM_UTC))
    return [Series(id="caiso_np15_rt_lmp_1h", values=rt.to_numpy(dtype=np.float64), step_s=3600.0, t0=rt.index[0],
                   group="caiso_rt", test_from=tf, cov=cov, cov_names=names)]


def uk_carbon_series(with_covariates: bool = True) -> list[Series]:
    """Great Britain actual carbon intensity, half-hourly averaged to hourly. Optional covariate: London day-ahead weather."""
    c = _hourly(P.load_carbon("gb", "actual")["intensity_g_per_kwh"], 2, 0.5)
    cov, names = None, ()
    if with_covariates:
        cov, names = _weather_cov("london", c.index)
    tf = int(c.index.searchsorted(TEST_FROM_UTC))
    return [Series(id="gb_carbon_actual_1h", values=c.to_numpy(dtype=np.float64), step_s=3600.0, t0=c.index[0],
                   group="gb_carbon", test_from=tf, cov=cov, cov_names=names)]


# --------------------------------------------------------------------------- simulator
def sim_cache_path(seed: int, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{SIM_TAG}_seed{seed}.npz"


def build_sim_cache(seed: int, path: Optional[Path] = None, horizon_s: Optional[int] = None) -> Path:
    """Run the configured sim scenario (SIM_SCENARIO) for one seed (SIMULATED data) and cache the arrays the forecasting studies use.

    Cached: rack_power_w (1 s, float32), rack_temp_c_1min, rack_max_gpu_temp_c_1min, site_mean_power_w_1h,
    per-GPU power in the fixed windows ``SIM_GPU_WINDOWS_S`` (float32), the known-future covariates of the
    rack 1-minute and site 1-hour grids, and the metadata json.
    """
    from gpu_stack.research import heatwave_sim as hs

    path = Path(path) if path else sim_cache_path(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    sc = hs.load_scenario(SIM_SCENARIO)
    if horizon_s is not None:
        sc["horizon_s"] = int(horizon_s)
    sc["outputs"] = {"gpu_windows": [list(w) for w in SIM_GPU_WINDOWS_S if w[1] <= sc["horizon_s"]]}
    a = hs.run_scenario(sc, seed=seed).to_arrays()
    keep = {
        "rack_power_w": a["rack_power_w"].astype(np.float32),
        "rack_temp_c_1min": a["rack_temp_c_1min"].astype(np.float32),
        "rack_max_gpu_temp_c_1min": a["rack_max_gpu_temp_c_1min"].astype(np.float32),
        "site_mean_power_w_1h": a["site_mean_power_w_1h"].astype(np.float32),
    }
    for i in range(len(sc["outputs"]["gpu_windows"])):
        keep[f"gpu_power_w_window_{i}"] = a[f"gpu_power_w_window_{i}"].astype(np.float32)
        keep[f"gpu_power_w_window_{i}_bounds_s"] = a[f"gpu_power_w_window_{i}_bounds_s"]
    for k, v in a.items():
        if k.startswith(("cov_rack_1min_", "cov_site_1h_")):
            keep[k] = np.asarray(v, dtype=np.float32)
    keep["meta_json"] = np.frombuffer(json.JSONEncoder(default=str).encode(a["meta"]).encode(), dtype=np.uint8)
    np.savez(path, **keep)
    return path


def load_sim_cache(seed: int, cache_dir: Path = CACHE_DIR) -> dict:
    z = np.load(sim_cache_path(seed, cache_dir))
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(bytes(out.pop("meta_json")).decode())
    return out


def sim_series(kind: str, seeds: Sequence[int] = SIM_TEST_SEEDS, cache_dir: Path = CACHE_DIR, with_covariates: bool = True) -> list[Series]:
    """Series from cached simulator runs (SIMULATED).

    kind: 'rack_power_1s' (rack_power_w, 1 s), 'rack_temp_1min' (mean rack air temperature, 1 min; known-future
    covariates: planned GPUs running, scheduled checkpoint GPUs, step phase, outdoor forecast, price, carbon),
    'site_power_1h' (site mean power per hour).
    """
    out = []
    for sd in seeds:
        z = load_sim_cache(sd, cache_dir)
        racks, sites = z["meta"]["rack_ids"], z["meta"]["site_ids"]
        if kind == "rack_power_1s":
            for r, rid in enumerate(racks):
                out.append(Series(id=f"seed{sd}/{rid}", values=z["rack_power_w"][:, r].astype(np.float64), step_s=1.0,
                                  t0=SIM_START_UTC, group=f"seed{sd}"))
        elif kind == "rack_temp_1min":
            cn = ("planned_gpus_running", "scheduled_checkpoint_gpus", "step_phase_compute_fraction",
                  "outdoor_forecast_c", "price_usd_per_kwh", "carbon_g_per_kwh")
            for r, rid in enumerate(racks):
                cov = None
                if with_covariates:
                    cov = np.vstack([z[f"cov_rack_1min_{n}"][:, r] for n in cn]).astype(np.float64)
                out.append(Series(id=f"seed{sd}/{rid}", values=z["rack_temp_c_1min"][:, r].astype(np.float64), step_s=60.0,
                                  t0=SIM_START_UTC, group=f"seed{sd}", cov=cov, cov_names=cn if with_covariates else ()))
        elif kind == "site_power_1h":
            for s, sid in enumerate(sites):
                out.append(Series(id=f"seed{sd}/{sid}", values=z["site_mean_power_w_1h"][:, s].astype(np.float64), step_s=3600.0,
                                  t0=SIM_START_UTC, group=f"seed{sd}"))
        else:
            raise ValueError(kind)
    return out


# --------------------------------------------------------------------------- dataset registry
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    scale: str
    data_kind: str
    quantity: str
    unit: str
    description: str
    build: Callable[[], list[Series]]
    stride: int
    max_per_series: Optional[int]


def dataset_specs(sim_seeds: Sequence[int] = SIM_TEST_SEEDS, mit: bool = True) -> dict[str, DatasetSpec]:
    """The datasets of the multi-scale study. Stride and thinning are fixed here (and recorded in the results)."""
    S = [
        DatasetSpec("laptop_power", "seconds", "measured", "GPU power (energy-counter)", "W",
                    "RTX 3060 Laptop GPU, measured run2 (TEST), 1 s", lambda: laptop_series("run2", "power", False), 60, None),
        DatasetSpec("elsayed_gpu_power", "seconds", "public", "GPU power", "W",
                    "Elsayed et al. 2026 8-GPU H100/B200 nodes, TEST sessions, per GPU, 1 s", lambda: elsayed_gpu_series("test", "power"), 86, 4),
        DatasetSpec("nlr_job_power", "seconds", "public", "multi-node job power", "W",
                    "NLR 2026 H100 training jobs, TEST runs, total power, 1 s", lambda: nlr_series("test"), 600, 10),
        DatasetSpec("sim_rack_power_1s", "seconds", "simulated", "rack power (SIMULATED)", "W",
                    f"gpu_stack heatwave simulator {SIM_TAG}, TEST seeds, rack_power_w, 1 s", lambda: sim_series("rack_power_1s", sim_seeds), 3600, 6),
        DatasetSpec("laptop_temp", "minutes", "measured", "GPU temperature", "deg C",
                    "RTX 3060 Laptop GPU, measured run2 (TEST), 15 s means (run is 79 min long)", lambda: laptop_series("run2", "temp", False), 6, None),
        DatasetSpec("sim_rack_temp_1min", "minutes", "simulated", "rack air temperature (SIMULATED)", "deg C",
                    f"gpu_stack heatwave simulator {SIM_TAG}, TEST seeds, rack_temp_c_1min", lambda: sim_series("rack_temp_1min", sim_seeds, with_covariates=False), 60, 6),
        DatasetSpec("caiso_rt_price_1h", "hours", "public", "CAISO RT LMP, hourly mean", "USD/MWh",
                    "CAISO NP15 real-time LMP, hourly means, origins from 2026-05-01", lambda: caiso_price_series(False), 12, None),
        DatasetSpec("uk_carbon_actual_1h", "hours", "public", "GB carbon intensity, hourly mean", "gCO2/kWh",
                    "National Energy System Operator actual intensity, hourly means, origins from 2026-05-01", lambda: uk_carbon_series(False), 12, None),
        DatasetSpec("sim_site_power_1h", "hours", "simulated", "site mean power (SIMULATED)", "W",
                    f"gpu_stack heatwave simulator {SIM_TAG}, TEST seeds, site_mean_power_w_1h", lambda: sim_series("site_power_1h", sim_seeds), 6, None),
    ]
    if mit:
        S.insert(3, DatasetSpec("mit_node_power", "seconds", "public", "node power (2 x V100)", "W",
                                "MIT Supercloud V100 nodes (local copy), TEST jobs, node_power_w, 1 s", lambda: mit_series("test", "power"), 1800, 3))
        S.insert(6, DatasetSpec("mit_node_temp", "minutes", "public", "node max GPU temperature", "deg C",
                                "MIT Supercloud V100 nodes (local copy), TEST jobs, node_temp_c_max, 1-minute means", lambda: mit_series("test", "temp"), 30, 4))
    return {s.name: s for s in S}


def build_dataset(spec: DatasetSpec) -> Windows:
    series = spec.build()
    w = make_windows(spec.name, spec.scale, spec.data_kind, series, stride=spec.stride, max_per_series=spec.max_per_series,
                     meta={"quantity": spec.quantity, "unit": spec.unit, "description": spec.description})
    return w


# --------------------------------------------------------------------------- hierarchies
def _hier_from_arrays(
    name: str, data_kind: str, hier: Hierarchy, arrays: Sequence[np.ndarray], ids: Sequence[str], groups: Sequence[str],
    *, context: int, horizon: int, stride: int, step_s: float, split: str, meta: dict,
) -> HierWindows:
    """arrays: list of (n, T_i) matrices (rows in hierarchy order). Cuts windows with a common origin rule."""
    cs, fs, wid, grp = [], [], [], []
    n_adm = 0
    for a, i, g in zip(arrays, ids, groups):
        pos, na = origin_positions(a.shape[1], context, horizon, stride)
        n_adm += na
        for o in pos:
            cs.append(a[:, o - context:o])
            fs.append(a[:, o:o + horizon])
            wid.append(f"{i}@{o}")
            grp.append(g)
    m = dict(meta)
    m.update({"n_windows": len(cs), "n_admissible_origins": n_adm, "context_steps": context, "horizon_steps": horizon, "stride_steps": stride})
    return HierWindows(
        name=name, data_kind=data_kind, step_s=step_s, season=1, hierarchy=hier,
        context=np.asarray(cs, dtype=np.float64), future=np.asarray(fs, dtype=np.float64),
        window_id=np.asarray(wid, dtype=object), group=np.asarray(grp, dtype=object), split=split, meta=m,
    )


def hier_public_node(split: str = "test", stride: int = 60, context: Optional[int] = None, horizon: Optional[int] = None) -> HierWindows:
    """GPU -> node (public): 8 GPU power series and the node total of an Elsayed et al. 2026 session, 1 s.
    Node power equals the sum of the GPU powers (max absolute difference below 0.001 W in the data)."""
    ctx = context or CONTEXT["seconds"]
    hor = horizon or HORIZON["seconds"]
    gpus = [f"g{k}" for k in range(8)]
    hier = build_tree("node", None, bottom_level="gpu", root_level="node", bottoms=gpus)
    arrays, ids, grp = [], [], []
    for sess in node_sessions(split):
        d = P.load_node_power(sess, "1s")
        g = np.vstack([d[f"{k}_power_w"].to_numpy(dtype=np.float64) for k in gpus])
        a = np.vstack([g.sum(axis=0, keepdims=True), g])  # node = exact sum of GPU rows
        arrays.append(a)
        ids.append(sess)
        grp.append(sess)
    return _hier_from_arrays(
        "hier_public_gpu_node", "public", hier, arrays, ids, grp, context=ctx, horizon=hor, stride=stride, step_s=1.0, split=split,
        meta={"description": "Elsayed et al. 2026 8-GPU nodes, node = sum of per-GPU power rows", "n_sessions": len(arrays)},
    )


def hier_sim(split: str = "test", seeds: Optional[Sequence[int]] = None, stride: int = 150, cache_dir: Path = CACHE_DIR,
             context: Optional[int] = None, horizon: Optional[int] = None) -> dict[str, HierWindows]:
    """GPU -> rack -> site of the SIMULATED cluster, per-GPU power at 1 s: split 'test' uses the evaluation windows (SIM_EVAL_WINDOWS), any other split the calibration windows.
    Returns one HierWindows per site (site-a: 4 racks, 104 GPUs; site-b: 4 racks, 80 GPUs). Rack = sum of its
    GPUs and site = sum of its racks (GPU power only; host and cooling power are not in this hierarchy)."""
    ctx = context or CONTEXT["seconds"]
    hor = horizon or HORIZON["seconds"]
    seeds = tuple(seeds if seeds is not None else (SIM_TEST_SEEDS if split == "test" else SIM_VAL_SEEDS))
    per_site: dict[str, tuple[list, list, list]] = {}
    hiers: dict[str, Hierarchy] = {}
    for sd in seeds:
        z = load_sim_cache(sd, cache_dir)
        meta = z["meta"]
        counts = meta["rack_gpu_counts"]
        edges = np.cumsum([0] + counts)
        rack_site = meta["rack_site_ids"]
        rack_ids = meta["rack_ids"]
        for sid in meta["site_ids"]:
            racks = [r for r, s in enumerate(rack_site) if s == sid]
            if sid not in hiers:
                mid = {rack_ids[r]: [f"{rack_ids[r]}.g{k}" for k in range(counts[r])] for r in racks}
                hiers[sid] = build_tree(sid, mid, bottom_level="gpu", root_level="site", mid_level="rack")
            for wi in (SIM_EVAL_WINDOWS if split == "test" else SIM_CALIB_WINDOWS):
                w = z[f"gpu_power_w_window_{wi}"].astype(np.float64)  # (184, 1800)
                g = np.vstack([w[edges[r]:edges[r + 1]] for r in racks])
                rack_rows = np.vstack([w[edges[r]:edges[r + 1]].sum(axis=0) for r in racks])
                a = np.vstack([rack_rows.sum(axis=0, keepdims=True), rack_rows, g])
                lst = per_site.setdefault(sid, ([], [], []))
                lst[0].append(a)
                lst[1].append(f"seed{sd}/win{wi}")
                lst[2].append(f"seed{sd}")
    out = {}
    for sid, (arrays, ids, grp) in per_site.items():
        out[sid] = _hier_from_arrays(
            f"hier_sim_gpu_rack_{sid}", "simulated", hiers[sid], arrays, ids, grp, context=ctx, horizon=hor, stride=stride,
            step_s=1.0, split=split,
            meta={"description": "SIMULATED gpu_stack heatwave simulator v2, per-GPU power windows; rack and site rows are exact sums",
                  "seeds": list(seeds),
                  "gpu_windows_s": [list(SIM_GPU_WINDOWS_S[i]) for i in (SIM_EVAL_WINDOWS if split == "test" else SIM_CALIB_WINDOWS)]},
        )
    return out
