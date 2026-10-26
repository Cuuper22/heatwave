"""Rolling forecast loop: Timestral and seasonal naive, side by side.

Every ``interval_s`` seconds: take the last ``context_s`` seconds of GPU power
and temperature (1 s bins, mean of the 10 Hz samples), build two known-future
covariates from the training load's checkpoint schedule (checkpoint active,
and checkpoint-cycle phase), forecast the next ``horizon_s`` seconds with both
forecasters, publish the 21 quantiles, and append the forecast to
``forecasts.jsonl`` so it can be scored against the logged actuals later.

Time grid: bin i covers [i, i+1) seconds of sampler time. A forecast whose
``origin_s`` is n uses bins 0..n-1 (the last ``context_s`` of them) and
predicts bins n .. n+horizon-1.
"""

from __future__ import annotations

import csv
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from heatwave.forecast import QUANTILE_LEVELS, QuantileForecast, SeasonalNaiveForecaster

from .state import GuardedThread, RunState, json_text
from .telemetry import Sampler, bin_series, iso_utc

VARIATES = ("power_w", "temp_c")


class ScheduleTracker:
    """Incrementally reads the load's events CSV and predicts checkpoints.

    Event times are converted to sampler time with ``start_unix``.
    """

    def __init__(self, events_path: str | Path, start_unix: float) -> None:
        self.path = Path(events_path)
        self.start_unix = start_unix
        self._lock = threading.Lock()
        self._offset = 0
        self._partial = ""
        self.starts: list[float] = []
        self.ends: list[float] = []
        self.step_times: deque = deque(maxlen=60)
        self.ckpt_every = 0
        self.last_step = 0

    def refresh(self) -> None:
        with self._lock:
            try:
                with open(self.path, "r", encoding="utf-8", newline="") as f:
                    f.seek(self._offset)
                    chunk = f.read()
                    self._offset = f.tell()
            except FileNotFoundError:
                return
            text = self._partial + chunk
            lines = text.split("\n")
            self._partial = lines.pop()  # incomplete tail, if any
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) != 5 or parts[0] == "t_utc":
                    continue
                try:
                    t = float(parts[1]) - self.start_unix
                    ev, step, val = parts[2], int(parts[3]), float(parts[4])
                except ValueError:
                    continue
                if ev == "config":
                    self.ckpt_every = int(val)
                elif ev == "step":
                    self.step_times.append(t)
                    self.last_step = step
                elif ev == "ckpt_start":
                    self.starts.append(t)
                elif ev == "ckpt_end":
                    self.ends.append(t)

    # --- schedule estimate -----------------------------------------------
    def duration(self) -> float:
        n = min(len(self.ends), len(self.starts))
        if n == 0:
            return 0.0
        d = [self.ends[i] - self.starts[i] for i in range(max(0, n - 5), n)]
        return float(np.mean(d))

    def period(self) -> Optional[float]:
        if len(self.starts) >= 2:
            return float(np.median(np.diff(self.starts[-6:])))
        if len(self.starts) == 1 and self.ckpt_every > 0 and len(self.step_times) >= 3:
            st = float(np.median(np.diff(list(self.step_times))))
            return self.ckpt_every * st + self.duration()
        return None

    def intervals(self, t_end: float) -> list[tuple[float, float]]:
        """Known checkpoint intervals plus predicted ones until ``t_end``."""
        with self._lock:
            starts = list(self.starts)
            ends = list(self.ends)
            period = self.period()
            dur = self.duration()
        out = []
        for i, s in enumerate(starts):
            e = ends[i] if i < len(ends) else s + dur
            out.append((s, e))
        if starts and period and period > 0:
            s = starts[-1] + period
            while s < t_end:
                out.append((s, s + dur))
                s += period
        return out

    def covariates(self, first_bin: int, origin: int, horizon: int) -> np.ndarray:
        """(2, T+H): checkpoint-active fraction per bin, checkpoint-cycle phase."""
        n = origin + horizon - first_bin
        active = np.zeros(n)
        phase = np.zeros(n)
        iv = self.intervals(origin + horizon + 1)
        with self._lock:
            period = self.period()
        starts = np.array([s for s, _ in iv])
        for k in range(n):
            lo, hi = first_bin + k, first_bin + k + 1
            for s, e in iv:
                ov = min(hi, e) - max(lo, s)
                if ov > 0:
                    active[k] += ov
            if period and len(starts):
                prior = starts[starts <= lo + 0.5]
                if len(prior):
                    phase[k] = min((lo + 0.5 - prior[-1]) / period, 1.5)
        return np.stack([np.clip(active, 0.0, 1.0), phase])


class ForecastRecord:
    def __init__(self, origin_s: int, horizon: int, issued_unix: float, issued_mono_s: float,
                 context_len: int, period_s: Optional[float]) -> None:
        self.origin_s = origin_s
        self.horizon = horizon
        self.issued_unix = issued_unix
        self.issued_mono_s = issued_mono_s
        self.context_len = context_len
        self.period_s = period_s
        self.forecasts: dict[str, QuantileForecast] = {}
        self.latency_s: dict[str, float] = {}

    def to_json(self) -> str:
        return json_text({
            "issued_utc": iso_utc(self.issued_unix),
            "issued_mono_s": round(self.issued_mono_s, 3),
            "origin_s": self.origin_s,
            "horizon": self.horizon,
            "step_s": 1.0,
            "context_len": self.context_len,
            "checkpoint_period_s": self.period_s,
            "levels": list(QUANTILE_LEVELS),
            "variates": list(VARIATES),
            "latency_s": {k: round(v, 4) for k, v in self.latency_s.items()},
            "forecasts": {
                name: {VARIATES[v]: np.round(fc.values[0, v], 3) for v in range(len(VARIATES))}
                for name, fc in self.forecasts.items()
            },
        })


class ForecastLoop(GuardedThread):
    def __init__(self, sampler: Sampler, tracker: ScheduleTracker, state: RunState,
                 jsonl_path: str | Path, stop_event: threading.Event, *,
                 forecaster=None, model_size: str = "5M", threads: int = 2,
                 interval_s: float = 5.0, horizon_s: int = 60, context_s: int = 512,
                 min_context_s: int = 20) -> None:
        super().__init__("forecast", stop_event)
        self.sampler = sampler
        self.tracker = tracker
        self.state = state
        self.jsonl_path = Path(jsonl_path)
        self.interval_s = interval_s
        self.horizon = int(horizon_s)
        self.context_s = int(context_s)
        self.min_context = int(min_context_s)
        self._forecaster = forecaster
        self.model_size = model_size
        self.threads = threads
        self.n_issued = 0

    def _build(self):
        if self._forecaster is None:
            from heatwave.forecast.timestral_fc import TimestralForecaster

            self._forecaster = TimestralForecaster(self.model_size, threads=self.threads)
        return self._forecaster

    def forecast_once(self) -> Optional[ForecastRecord]:
        n = int(self.sampler.elapsed_s())
        if n < self.min_context:
            return None
        first = max(0, n - self.context_s)
        snap = self.sampler.window(n - first + 1.0)
        if not snap:
            return None
        t = np.array([s.t_mono_s - first for s in snap])
        ctx = np.stack([
            bin_series(t, np.array([s.values["power_w"] for s in snap]), n - first),
            bin_series(t, np.array([s.values["temp_c"] for s in snap]), n - first),
        ])
        if np.isnan(ctx).all(axis=1).any():
            return None
        self.tracker.refresh()
        cov = self.tracker.covariates(first, n, self.horizon)
        period = self.tracker.period()
        rec = ForecastRecord(n, self.horizon, time.time(), self.sampler.elapsed_s(),
                             ctx.shape[1], period)
        tf = self._build()
        t0 = time.perf_counter()
        rec.forecasts[tf.name] = tf.forecast(ctx[None], self.horizon, known_covariates=cov[None])
        rec.latency_s[tf.name] = time.perf_counter() - t0
        season = int(round(period)) if period else 1
        sn = SeasonalNaiveForecaster(max(1, season))
        t0 = time.perf_counter()
        # Stable key; the season (checkpoint period, or 1 = persistence before
        # one is known) is recorded in the record's checkpoint_period_s.
        rec.forecasts["seasonal-naive"] = sn.forecast(ctx[None], self.horizon)
        rec.latency_s["seasonal-naive"] = time.perf_counter() - t0
        return rec

    def loop(self) -> None:
        self._build()
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            while not self.stop_event.is_set():
                rec = self.forecast_once()
                if rec is not None:
                    self.state.set_forecast(rec)
                    f.write(rec.to_json() + "\n")
                    f.flush()
                    self.n_issued += 1
                self.stop_event.wait(self.interval_s)
