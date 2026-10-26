"""Known-future covariate builders with explicit leakage rules.

A covariate may only carry information available at the forecast origin for its horizon step: forecasts, published
prices and pre-written schedules. It must never carry a realised future actual. Each builder here returns, for one
forecast origin, an array over the context window and the horizon, and enforces its own availability rule:

  * the sim builders (SIMULATED data) take the planned job schedule from jobs submitted at or before the origin only,
    pass through the simulated outdoor forecast, and limit the horizon to the scenario's known-future horizon;
  * the day-ahead price builder reveals a CAISO day-ahead price only after its publication time;
  * the archived-forecast builder reveals a forecast value only if its issue time is at or before the origin;
  * the laptop schedule builders use the pre-written load schedule; the checkpoint covariate projects future
    checkpoints from the last checkpoint seen before the origin (a variant that reads the logged checkpoint
    intervals of the whole run is kept separate and labelled).

Every builder has a test in tests/test_covariates.py that changes the realised future (or the unavailable source
values) and asserts that the covariate does not change.

Shapes follow `heatwave.forecast.base`: covariate arrays are (C, T + H) with T context steps and H horizon steps; NaN
marks a value that is not available at the origin (Timestral treats NaN as missing).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SCENARIO_PATH = os.environ.get(
    "HEATWAVE_SIM_SCENARIO", "/home/claude/gpu_stack-/experiments/e003-heatwave/sim-scenario-v2.json"
)


class LeakageError(ValueError):
    """Raised when a request would reveal information not available at the forecast origin."""


# --------------------------------------------------------------------------------------- simulator (SIMULATED data)
#: covariate groups of the simulator and the planned_covariates keys each one uses
SIM_GROUPS: dict[str, tuple[str, ...]] = {
    "J": ("planned_gpus_running", "scheduled_checkpoint_gpus"),  # job schedule
    "P": ("step_phase_compute_fraction",),  # training-step phase
    "W": ("outdoor_forecast_c",),  # weather (outdoor temperature) forecast
    "C": ("price_usd_per_kwh", "carbon_g_per_kwh"),  # price and carbon forecast
}
SIM_GROUP_NAMES = {
    "J": "job schedule", "P": "training-step phase", "W": "weather forecast", "C": "price/carbon forecast",
}


def world_scenario(scenario: Mapping, seed: int, variant: str = "worlds") -> dict:
    """Scenario variant for run seed ``seed`` (SIMULATED).

    ``literal``: the committed scenario unchanged. Its job-generator seed (7) and environment seed (11) are fixed in
    the file, so the run seed only changes the noise streams and every seed shares one job schedule and weather path.
    ``worlds``: the job-generator seed and the environment seed are set to ``seed``, so each seed is a different
    synthetic job schedule, weather, price and carbon path. Fault and grid events keep their scheduled times.
    """
    sc = copy.deepcopy(dict(scenario))
    if variant == "literal":
        return sc
    if variant != "worlds":
        raise ValueError("variant must be 'literal' or 'worlds'")
    sc["jobs"]["generate"]["seed"] = int(seed)
    sc["environment"]["seed"] = int(seed)
    return sc


class SimCovariateBuilder:
    """Known-future covariates of the simulator for one (scenario variant, seed), built per forecast origin.

    ``origin_s`` is the simulated time (seconds) of the first forecast step. Leakage rules:
      * jobs enter the plan only if ``submit_s <= origin_s`` (a job submitted later is unknown at the origin). The plan
        before the origin is unchanged by jobs that start after it, so the context part is the same plan;
      * the horizon may not exceed the scenario's ``known_future_horizon_s``;
      * the weather covariate is the scenario's outdoor *forecast* series (``SiteEnvironment.forecast_c``), never the
        actual outdoor temperature. In the simulator the forecast is the simulated outdoor temperature plus an AR(1)
        forecast error (sigma 1.2 degC), so it holds the realised noise of the simulated weather: a simulated
        forecast, not a real one. Price and carbon are passed through without error (day-ahead style).
    """

    def __init__(self, scenario: Mapping, seed: int, variant: str = "worlds") -> None:
        from gpu_stack.research import heatwave_sim as hs

        self._hs = hs
        self.variant = variant
        self.seed = int(seed)
        self.scenario = world_scenario(scenario, seed, variant)
        self.cfg = hs.build_config(self.scenario, seed=self.seed)
        self.known_future_horizon_s = float(self.cfg.known_future_horizon_s)
        self.rack_ids = [r.rack_id for lay in self.cfg.layouts for r in lay.racks]
        self.site_ids = [lay.site.site_id for lay in self.cfg.layouts]
        self._jobs = sorted(self.cfg.jobs, key=lambda j: j.submit_s)
        self._submit = np.array([j.submit_s for j in self._jobs])
        self._plans: dict[int, tuple] = {}

    def world_key(self) -> tuple:
        """Identity of the job schedule and environment (equal for all seeds of the 'literal' variant)."""
        g = self.scenario["jobs"]["generate"]["seed"]
        return (g, self.scenario["environment"].get("seed"))

    def n_known_jobs(self, origin_s: float) -> int:
        return int(np.searchsorted(self._submit, origin_s, side="right"))

    def _plan(self, n_jobs: int):
        if n_jobs not in self._plans:
            cfg = replace(self.cfg, jobs=tuple(self._jobs[:n_jobs]))
            self._plans[n_jobs] = (cfg, self._hs.plan_schedule(cfg))
        return self._plans[n_jobs]

    def window(
        self, level: str, resolution_s: float, origin_idx: int, context: int, horizon: int,
        *, full_knowledge: bool = False,
    ) -> dict[str, np.ndarray]:
        """Covariates for bins [origin_idx - context, origin_idx + horizon) of ``resolution_s`` seconds.

        Returns {group: (C_g, units, context + horizon)}. ``full_knowledge=True`` plans with every job in the scenario,
        including jobs submitted after the origin; it exists only so a test can show the difference.
        """
        if horizon * resolution_s > self.known_future_horizon_s + 1e-9:
            raise LeakageError(
                f"horizon {horizon * resolution_s:.0f} s exceeds the known-future horizon {self.known_future_horizon_s:.0f} s"
            )
        origin_s = origin_idx * resolution_s
        t0 = (origin_idx - context) * resolution_s
        t1 = (origin_idx + horizon) * resolution_s
        if t0 < 0 or t1 > self.cfg.horizon_s + 1e-9:
            raise ValueError("window outside the simulated span")
        n = len(self._jobs) if full_knowledge else self.n_known_jobs(origin_s)
        cfg, plans = self._plan(n)
        cov = self._hs.planned_covariates(cfg, plans, resolution_s, level, t0, t1)
        out = {}
        for g, keys in SIM_GROUPS.items():
            out[g] = np.stack([cov[k].T for k in keys], axis=0)  # (C, units, T+H)
        return out


# ------------------------------------------------------------------------------- real data: issue-time covariates
class IssueTimeCovariate:
    """Covariate values on a regular forecast grid, each with the earliest origin at which it is known.

    ``values`` is (N, C) aligned to the grid, ``available_from`` is (N,) int64 nanoseconds (UTC): the value for grid
    step ``s`` may be used at an origin ``o`` only if ``available_from[s] <= o``. Steps before the origin are always
    available (``available_from`` is capped at the step's own time). Unavailable values are NaN.
    """

    def __init__(self, grid: pd.DatetimeIndex, values: np.ndarray, available_from: np.ndarray, label: str,
                 vintage_verified: bool = True, note: str = "") -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if len(grid) != len(values) or len(grid) != len(available_from):
            raise ValueError("grid, values and available_from must have the same length")
        self.grid = grid
        self.grid_ns = grid.asi8
        self.values = values
        self.available_from = np.minimum(np.asarray(available_from, dtype=np.int64), self.grid_ns)
        self.label = label
        self.vintage_verified = vintage_verified
        self.note = note

    @property
    def n_channels(self) -> int:
        return self.values.shape[1]

    def window(self, origin_idx: int, context: int, horizon: int) -> np.ndarray:
        """(C, context + horizon) for grid steps [origin_idx - context, origin_idx + horizon)."""
        lo, hi = origin_idx - context, origin_idx + horizon
        if lo < 0 or hi > len(self.grid):
            raise ValueError("window outside the grid")
        origin_ns = self.grid_ns[origin_idx]
        v = self.values[lo:hi].copy()
        v[self.available_from[lo:hi] > origin_ns] = np.nan
        return v.T

    def availability_end(self, origin_idx: int, horizon: int) -> int:
        """Number of horizon steps (from the origin) whose value is available (prefix length)."""
        origin_ns = self.grid_ns[origin_idx]
        a = self.available_from[origin_idx:origin_idx + horizon] <= origin_ns
        return int(a.argmin()) if not a.all() else int(len(a))


def _floor_to_source(grid: pd.DatetimeIndex, src_index: pd.DatetimeIndex) -> np.ndarray:
    """For each grid step the position of the latest source label at or before it (-1 if none)."""
    pos = np.searchsorted(src_index.asi8, grid.asi8, side="right") - 1
    return pos


def publication_instants(grid: pd.DatetimeIndex, *, tz: str = "America/Los_Angeles", publish_local_hour: int = 13) -> np.ndarray:
    """For each grid step, the UTC instant (ns) at which day-ahead products for its local day are published.

    The day-ahead products for local calendar day D are published at ``publish_local_hour`` local time on day D - 1.
    """
    local_day = grid.tz_convert(tz).normalize()
    days = local_day.unique()
    pub = pd.DatetimeIndex(
        [(d.tz_localize(None) - pd.Timedelta("1D") + pd.Timedelta(f"{publish_local_hour}h")).tz_localize(
            tz, ambiguous=True, nonexistent="shift_forward") for d in days]
    ).tz_convert("UTC")
    day_to_pub = dict(zip(days, pub.asi8))
    return np.array([day_to_pub[d] for d in local_day], dtype=np.int64)


def day_ahead_price_covariate(
    dam: pd.Series, grid: pd.DatetimeIndex, *, tz: str = "America/Los_Angeles", publish_local_hour: int = 13,
    label: str = "caiso_dam_price",
) -> IssueTimeCovariate:
    """CAISO day-ahead LMP (hourly, interval start, UTC) held over a finer forecast grid.

    Publication rule: the day-ahead prices for a local calendar day are published at ``publish_local_hour`` local time
    on the day before (CAISO publishes day-ahead market results in the early afternoon, Pacific time). A step on local
    day D is therefore available from (D - 1) at ``publish_local_hour``. Assumption: 13:00 Pacific; the real
    publication time varies by a few minutes.
    """
    pos = _floor_to_source(grid, dam.index)
    ok = pos >= 0
    vals = np.full(len(grid), np.nan)
    vals[ok] = dam.to_numpy(dtype=float)[pos[ok]]
    avail = publication_instants(grid, tz=tz, publish_local_hour=publish_local_hour)
    return IssueTimeCovariate(
        grid, vals, avail, label,
        note=f"day-ahead price, available from {publish_local_hour}:00 {tz} on the day before delivery (assumed publication time)",
    )


def archived_forecast_covariate(
    fc: pd.DataFrame, grid: pd.DatetimeIndex, columns: Sequence[str], *, issue_lead: pd.Timedelta, label: str,
    vintage_verified: bool = True, note: str = "",
) -> IssueTimeCovariate:
    """An archived forecast series held over the forecast grid; the value for time v is known from v - issue_lead.

    ``issue_lead`` is the lead of the vintage (previous_day1 weather: 24 h). For vintages that are not documented
    (GB carbon forecast) pass ``vintage_verified=False`` and the lead that is assumed; the result is then conditional on
    that assumption.
    """
    pos = _floor_to_source(grid, fc.index)
    ok = pos >= 0
    vals = np.full((len(grid), len(columns)), np.nan)
    vals[ok] = fc[list(columns)].to_numpy(dtype=float)[pos[ok]]
    label_ns = fc.index.asi8[np.where(ok, pos, 0)]
    avail = label_ns - int(issue_lead.value)
    return IssueTimeCovariate(grid, vals, avail, label, vintage_verified=vintage_verified, note=note)


# ------------------------------------------------------------------------------------ laptop (MEASURED data) schedule
@dataclass(frozen=True)
class LaptopSchedule:
    """The pre-written load schedule of a laptop run (``scripts/laptop_collect.py``) plus what has been observed.

    ``segments``: [(name, seconds, mode, batch, ckpt_every_s)] as in the run's meta. ``seg_start``: observed start
    time (UTC ns) of each segment (segment_start events). ``ckpt_start``/``ckpt_end``: observed checkpoint event times
    (UTC ns). ``ckpt_d_s``: nominal checkpoint write duration (median over a different run, see the experiment script).
    """

    segments: tuple
    seg_start: np.ndarray
    ckpt_start: np.ndarray
    ckpt_end: np.ndarray
    ckpt_d_s: float


def laptop_schedule_from_events(segments: Sequence, events: pd.DataFrame, ckpt_d_s: float) -> LaptopSchedule:
    ss = events[events.kind == "segment_start"]
    seg_start = ss.index.asi8[: len(segments)]
    return LaptopSchedule(
        tuple(tuple(s) for s in segments), np.asarray(seg_start, dtype=np.int64),
        events[events.kind == "ckpt_start"].index.asi8.copy(), events[events.kind == "ckpt_end"].index.asi8.copy(),
        float(ckpt_d_s),
    )


def _interval_fraction(grid_ns: np.ndarray, res_ns: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Fraction of each grid bin [g, g + res) covered by the union of non-overlapping intervals."""
    out = np.zeros(len(grid_ns))
    for a, b in zip(starts, ends):
        lo = np.searchsorted(grid_ns + res_ns, a, side="right")
        hi = np.searchsorted(grid_ns, b, side="left")
        for j in range(lo, min(hi, len(grid_ns))):
            ov = min(grid_ns[j] + res_ns, b) - max(grid_ns[j], a)
            if ov > 0:
                out[j] += ov / res_ns
    return np.clip(out, 0.0, 1.0)


def laptop_training_on_future(
    sched: LaptopSchedule, grid: pd.DatetimeIndex, res_s: float, origin_idx: int, horizon: int, *, sub: int = 10,
) -> np.ndarray:
    """Scheduled load-active indicator (bin mean) for horizon steps, from the schedule and the observed start of the current segment.

    Uses only the pre-written segment list and the observed start time of the segment that is running at the origin.
    """
    origin_ns = grid.asi8[origin_idx]
    res_ns = int(round(res_s * 1e9))
    k = int(np.searchsorted(sched.seg_start, origin_ns, side="right") - 1)
    if k < 0:
        raise LeakageError("origin before the first segment start")
    # nominal boundaries from the observed start of segment k
    secs = np.array([s[1] for s in sched.segments], dtype=float)
    b = sched.seg_start[k] + np.concatenate([[0.0], np.cumsum(secs[k:])]) * 1e9
    t = grid.asi8[origin_idx] + (np.arange(horizon * sub) + 0.5) * res_ns / sub
    seg = np.searchsorted(b, t, side="right") - 1
    on = np.zeros(len(t))
    for j in np.unique(seg):
        if j < 0 or j >= len(secs[k:]):
            continue
        m = seg == j
        mode = sched.segments[k + j][2]
        if mode == "train":
            on[m] = 1.0
        elif mode == "square":
            el = (t[m] - b[j]) / 1e9
            on[m] = ((el // 60) % 2 == 0).astype(float)
    return on.reshape(horizon, sub).mean(axis=1)


def laptop_ckpt_starts_future(
    sched: LaptopSchedule, origin_ns: int, horizon_ns: int,
) -> list[int]:
    """Projected checkpoint start times (UTC ns) in (origin, origin + horizon]; see ``laptop_ckpt_future``."""
    end_ns = origin_ns + horizon_ns
    d_ns = int(sched.ckpt_d_s * 1e9)
    k = int(np.searchsorted(sched.seg_start, origin_ns, side="right") - 1)
    if k < 0:
        raise LeakageError("origin before the first segment start")
    secs = np.array([s[1] for s in sched.segments], dtype=float)
    seg_b = sched.seg_start[k] + np.concatenate([[0.0], np.cumsum(secs[k:])]) * 1e9
    starts = []
    for j in range(len(secs) - k):
        name, _, mode, _, ck = sched.segments[k + j]
        if mode != "train" or not ck:
            continue
        s0, s1 = int(seg_b[j]), int(seg_b[j + 1])
        if s1 <= origin_ns:
            continue
        ref = s0
        if j == 0:
            seen = sched.ckpt_start[(sched.ckpt_start >= s0) & (sched.ckpt_start < origin_ns)]
            if len(seen):
                last = int(seen[-1])
                e_seen = sched.ckpt_end[(sched.ckpt_end >= last) & (sched.ckpt_end < origin_ns)]
                ref = int(e_seen[0]) if len(e_seen) else last + d_ns
        t = ref + int(ck * 1e9)
        while t < s1 and t < end_ns:
            if t + d_ns > origin_ns:
                starts.append(t)
            t += d_ns + int(ck * 1e9)
    return starts


def laptop_ckpt_future(
    sched: LaptopSchedule, grid: pd.DatetimeIndex, res_s: float, origin_idx: int, horizon: int,
) -> np.ndarray:
    """Projected checkpoint indicator (bin fraction) for horizon steps.

    Uses the segment list (checkpoint period per segment), the observed checkpoints before the origin and a nominal
    write duration: the next checkpoint starts one period after the end of the last observed one (or after the segment
    start), then every period plus a write duration, until the nominal end of each segment.
    """
    origin_ns = int(grid.asi8[origin_idx])
    res_ns = int(round(res_s * 1e9))
    d_ns = int(sched.ckpt_d_s * 1e9)
    starts = np.array(laptop_ckpt_starts_future(sched, origin_ns, horizon * res_ns), dtype=np.int64)
    g = grid.asi8[origin_idx:origin_idx + horizon]
    return _interval_fraction(g, res_ns, starts, starts + d_ns)
