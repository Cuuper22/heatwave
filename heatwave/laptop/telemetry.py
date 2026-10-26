"""GPU telemetry sampler with NVML, nvidia-smi and simulated backends.

Backends all expose ``info()`` (limits, names), ``read()`` (one sample as a
dict keyed by SAMPLE_FIELDS, NaN where a value is not available) and
``close()``.

* NvmlBackend: ``pynvml`` / ``nvidia-ml-py`` (pip install nvidia-ml-py).
* NvidiaSmiBackend: streams ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits -lms N``.
* FakeBackend: SIMULATED. A first-order RC thermal model driven by an activity
  signal; used for tests and for end-to-end runs on machines without a GPU.
  Every row it produces carries data_kind="simulated".

The Sampler writes a CSV with a UTC ISO timestamp and monotonic seconds since
sampler start, at 10 Hz by default, flushed every second.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .state import GuardedThread

NAN = float("nan")

SAMPLE_FIELDS = (
    "power_w", "energy_j", "temp_c", "sm_clock_mhz", "mem_clock_mhz",
    "util_gpu_pct", "util_mem_pct", "fan_pct", "power_limit_w",
    "enforced_limit_w", "mem_used_mib",
)
CSV_COLUMNS = ("t_utc", "t_mono_s") + SAMPLE_FIELDS + ("data_kind",)


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class GpuInfo:
    name: str
    index: int
    default_limit_w: float
    min_limit_w: float
    max_limit_w: float
    clock_max_mhz: float = NAN
    clock_min_mhz: float = NAN
    power_limit_w: float = NAN  # current limit; NaN when the driver reports N/A
    backend: str = ""
    data_kind: str = "measured"


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


def parse_smi_number(text: str) -> float:
    """'[N/A]', '[Not Supported]', '' -> NaN; otherwise float."""
    s = text.strip()
    if not s or s.startswith("["):
        return NAN
    if s.upper() in ("N/A", "NOT SUPPORTED"):
        return NAN
    return _num(s)


# --------------------------------------------------------------------------
# NVML
# --------------------------------------------------------------------------
class NvmlBackend:
    name = "nvml"
    data_kind = "measured"

    def __init__(self, index: int = 0, nvml=None) -> None:
        if nvml is None:
            try:
                import pynvml as nvml  # provided by nvidia-ml-py
            except ImportError as e:
                raise BackendUnavailable("pynvml / nvidia-ml-py is not installed") from e
        self._n = nvml
        try:
            nvml.nvmlInit()
            self._h = nvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as e:  # NVMLError subclasses vary by version
            raise BackendUnavailable(f"NVML init failed: {e}") from e
        self.index = index

    def _try(self, fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    def info(self) -> GpuInfo:
        n, h = self._n, self._h
        name = self._try(n.nvmlDeviceGetName, h)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        cons = self._try(n.nvmlDeviceGetPowerManagementLimitConstraints, h)
        dflt = self._try(n.nvmlDeviceGetPowerManagementDefaultLimit, h)
        cur = self._try(n.nvmlDeviceGetPowerManagementLimit, h)
        cmax = self._try(n.nvmlDeviceGetMaxClockInfo, h, n.NVML_CLOCK_SM)
        cmin = NAN
        mem_clocks = self._try(n.nvmlDeviceGetSupportedMemoryClocks, h)
        if mem_clocks:
            gr = self._try(n.nvmlDeviceGetSupportedGraphicsClocks, h, mem_clocks[0])
            if gr:
                cmin = float(min(gr))
        return GpuInfo(
            name=name or "unknown GPU", index=self.index,
            default_limit_w=dflt / 1000.0 if dflt else NAN,
            min_limit_w=cons[0] / 1000.0 if cons else NAN,
            max_limit_w=cons[1] / 1000.0 if cons else NAN,
            clock_max_mhz=float(cmax) if cmax else NAN, clock_min_mhz=cmin,
            power_limit_w=cur / 1000.0 if cur else NAN,
            backend=self.name, data_kind=self.data_kind,
        )

    def read(self) -> dict:
        n, h = self._n, self._h
        out = {k: NAN for k in SAMPLE_FIELDS}
        v = self._try(n.nvmlDeviceGetPowerUsage, h)
        if v is not None:
            out["power_w"] = v / 1000.0
        v = self._try(n.nvmlDeviceGetTotalEnergyConsumption, h)
        if v is not None:
            out["energy_j"] = v / 1000.0
        v = self._try(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU)
        if v is not None:
            out["temp_c"] = float(v)
        v = self._try(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_SM)
        if v is not None:
            out["sm_clock_mhz"] = float(v)
        v = self._try(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_MEM)
        if v is not None:
            out["mem_clock_mhz"] = float(v)
        u = self._try(n.nvmlDeviceGetUtilizationRates, h)
        if u is not None:
            out["util_gpu_pct"] = float(u.gpu)
            out["util_mem_pct"] = float(u.memory)
        v = self._try(n.nvmlDeviceGetFanSpeed, h)
        if v is not None:
            out["fan_pct"] = float(v)
        v = self._try(n.nvmlDeviceGetPowerManagementLimit, h)
        if v is not None:
            out["power_limit_w"] = v / 1000.0
        v = self._try(n.nvmlDeviceGetEnforcedPowerLimit, h)
        if v is not None:
            out["enforced_limit_w"] = v / 1000.0
        m = self._try(n.nvmlDeviceGetMemoryInfo, h)
        if m is not None:
            out["mem_used_mib"] = m.used / 2**20
        return out

    def close(self) -> None:
        try:
            self._n.nvmlShutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------
# nvidia-smi streaming
# --------------------------------------------------------------------------
SMI_STREAM_FIELDS = (
    ("power.draw", "power_w"), ("temperature.gpu", "temp_c"),
    ("clocks.sm", "sm_clock_mhz"), ("clocks.mem", "mem_clock_mhz"),
    ("utilization.gpu", "util_gpu_pct"), ("utilization.memory", "util_mem_pct"),
    ("fan.speed", "fan_pct"), ("power.limit", "power_limit_w"),
    ("enforced.power.limit", "enforced_limit_w"), ("memory.used", "mem_used_mib"),
)
SMI_INFO_FIELDS = (
    "name", "power.default_limit", "power.min_limit", "power.max_limit",
    "power.limit", "clocks.max.sm",
)


def parse_smi_stream_line(line: str) -> dict:
    row = next(csv.reader([line]))
    out = {k: NAN for k in SAMPLE_FIELDS}
    for (_, key), cell in zip(SMI_STREAM_FIELDS, row):
        out[key] = parse_smi_number(cell)
    return out


class _SmiReader(threading.Thread):
    def __init__(self, cmd: Sequence[str]) -> None:
        super().__init__(name="smi-reader", daemon=True)
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, **kwargs,
        )
        self.latest: Optional[dict] = None
        self._lock = threading.Lock()
        self.stopping = False

    def run(self) -> None:
        try:
            for line in self.proc.stdout:
                if line.strip():
                    try:
                        parsed = parse_smi_stream_line(line)
                    except Exception:
                        continue
                    with self._lock:
                        self.latest = parsed
        except (ValueError, OSError):
            pass

    def get(self) -> Optional[dict]:
        with self._lock:
            return dict(self.latest) if self.latest else None

    def stop(self) -> None:
        self.stopping = True
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            pass
        sig = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
        try:
            self.proc.send_signal(sig)
            self.proc.wait(timeout=5)
        except Exception:
            print("[telemetry] nvidia-smi stream process did not stop; end it manually "
                  f"(pid {self.proc.pid})", flush=True)


class NvidiaSmiBackend:
    name = "nvidia-smi"
    data_kind = "measured"

    def __init__(self, index: int = 0, interval_ms: int = 100,
                 smi: str = "nvidia-smi",
                 stream_cmd: Optional[Sequence[str]] = None,
                 info_cmd: Optional[Sequence[str]] = None) -> None:
        if stream_cmd is None and shutil.which(smi) is None:
            raise BackendUnavailable("nvidia-smi was not found on PATH")
        self.index = index
        self._smi = smi
        self._info_cmd = list(info_cmd) if info_cmd else [
            smi, "-i", str(index),
            "--query-gpu=" + ",".join(SMI_INFO_FIELDS),
            "--format=csv,noheader,nounits",
        ]
        cmd = list(stream_cmd) if stream_cmd else [
            smi, "-i", str(index),
            "--query-gpu=" + ",".join(f for f, _ in SMI_STREAM_FIELDS),
            "--format=csv,noheader,nounits", "-lms", str(interval_ms),
        ]
        self._reader = _SmiReader(cmd)
        self._reader.start()

    def info(self) -> GpuInfo:
        try:
            res = subprocess.run(self._info_cmd, capture_output=True, text=True, timeout=20)
            row = next(csv.reader([res.stdout.strip().splitlines()[0]]))
            row = [c.strip() for c in row]
        except Exception as e:
            raise BackendUnavailable(f"could not read GPU info from nvidia-smi: {e}") from e
        cmin = NAN
        try:
            sc = subprocess.run(
                [self._smi, "-i", str(self.index), "--query-supported-clocks=gr",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20)
            vals = [parse_smi_number(x) for x in sc.stdout.split()]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                cmin = min(vals)
        except Exception:
            pass
        return GpuInfo(
            name=row[0], index=self.index,
            default_limit_w=parse_smi_number(row[1]), min_limit_w=parse_smi_number(row[2]),
            max_limit_w=parse_smi_number(row[3]), power_limit_w=parse_smi_number(row[4]),
            clock_max_mhz=parse_smi_number(row[5]), clock_min_mhz=cmin,
            backend=self.name, data_kind=self.data_kind,
        )

    def read(self) -> dict:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            got = self._reader.get()
            if got is not None:
                return got
            time.sleep(0.02)
        return {k: NAN for k in SAMPLE_FIELDS}

    def close(self) -> None:
        self._reader.stop()


# --------------------------------------------------------------------------
# Simulated backend
# --------------------------------------------------------------------------
@dataclass
class FakeParams:
    """All values are SIMULATED and arbitrary except where noted.

    default/min/max power limits (80/1/105 W) mirror what `nvidia-smi` reported
    on the lead's laptop, RTX 3060 Laptop GPU (checked by the project lead on
    2026-10-26). Everything else (thermal resistance, capacitance, clocks,
    noise) is made up for tests.
    """

    ambient_c: float = 28.0
    start_temp_c: float = 55.0
    r_k_per_w: float = 0.75
    c_j_per_k: float = 60.0
    idle_w: float = 15.0
    load_w: float = 95.0
    default_limit_w: float = 80.0
    min_limit_w: float = 1.0
    max_limit_w: float = 105.0
    clock_max_mhz: float = 1700.0
    clock_min_mhz: float = 210.0
    power_noise_w: float = 1.5
    temp_noise_c: float = 0.15
    seed: int = 0
    # "clock": -pl is rejected as unsupported, -lgc works (like the laptop);
    # "power": -pl and -lgc work; "noadmin": every change is refused.
    smi_mode: str = "clock"


class FakeBackend:
    name = "fake"
    data_kind = "simulated"

    def __init__(self, params: Optional[FakeParams] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.p = params or FakeParams()
        self._clock = clock
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.p.seed)
        self.temp_c = self.p.start_temp_c
        self.activity = 0.0
        self.power_w = self.p.idle_w
        self.energy_j = 0.0
        self.sm_clock = self.p.clock_min_mhz
        self._limit_w: Optional[float] = None  # None = driver default
        self._clock_cap: Optional[float] = None
        self._t_last = self._clock()
        self.smi_log: list[list[str]] = []

    # --- load interface -------------------------------------------------
    def set_activity(self, a: float) -> None:
        with self._lock:
            self._advance()
            self.activity = float(a)

    # --- physics --------------------------------------------------------
    def _cap_frac(self) -> float:
        if self._clock_cap is None:
            return 1.0
        return max(self._clock_cap / self.p.clock_max_mhz, 0.0)

    def _advance(self) -> None:
        now = self._clock()
        remaining = now - self._t_last
        self._t_last = now
        p = self.p
        while remaining > 1e-9:
            dt = min(remaining, 0.1)
            remaining -= dt
            dyn_demand = self.activity * (p.load_w - p.idle_w) * self._cap_frac() ** 2
            demand = p.idle_w + dyn_demand
            limit = self._limit_w if self._limit_w is not None else p.default_limit_w
            self.power_w = min(demand, limit)
            dyn = max(self.power_w - p.idle_w, 0.0)
            frac = math.sqrt(dyn / (p.load_w - p.idle_w)) if p.load_w > p.idle_w else 0.0
            frac = min(max(frac, p.clock_min_mhz / p.clock_max_mhz), self._cap_frac())
            self.sm_clock = p.clock_max_mhz * frac
            self.temp_c += dt * (self.power_w - (self.temp_c - p.ambient_c) / p.r_k_per_w) / p.c_j_per_k
            self.energy_j += self.power_w * dt

    # --- backend interface ---------------------------------------------
    def info(self) -> GpuInfo:
        p = self.p
        return GpuInfo(
            name="SIMULATED GPU", index=0, default_limit_w=p.default_limit_w,
            min_limit_w=p.min_limit_w, max_limit_w=p.max_limit_w,
            clock_max_mhz=p.clock_max_mhz, clock_min_mhz=p.clock_min_mhz,
            power_limit_w=NAN if p.smi_mode == "clock" else p.default_limit_w,
            backend=self.name, data_kind=self.data_kind,
        )

    def read(self) -> dict:
        with self._lock:
            self._advance()
            p = self.p
            limit = self._limit_w if self._limit_w is not None else p.default_limit_w
            out = {k: NAN for k in SAMPLE_FIELDS}
            out["power_w"] = max(self.power_w + self._rng.normal(0, p.power_noise_w), 0.0)
            out["energy_j"] = self.energy_j
            out["temp_c"] = self.temp_c + self._rng.normal(0, p.temp_noise_c)
            out["sm_clock_mhz"] = self.sm_clock
            out["mem_clock_mhz"] = 7000.0
            out["util_gpu_pct"] = 100.0 * min(1.0, self.activity)
            out["util_mem_pct"] = 40.0 * min(1.0, self.activity)
            out["power_limit_w"] = NAN if p.smi_mode == "clock" else limit
            out["enforced_limit_w"] = limit
            out["mem_used_mib"] = 2048.0
            return out

    def close(self) -> None:
        pass

    # --- emulated nvidia-smi (used by the controller's runner) ---------
    def smi(self, args: Sequence[str]) -> tuple[int, str, str]:
        """Emulate the few nvidia-smi calls the controller makes. The message
        wording is approximate; the controller matches on keywords only."""
        args = list(args)
        self.smi_log.append(args)
        p = self.p
        with self._lock:
            self._advance()
            if p.smi_mode == "noadmin":
                return 1, "", "Insufficient Permissions\n"
            if "-pl" in args:
                if p.smi_mode == "clock":
                    return 1, "", "Unable to set power management limit: Not Supported\n"
                w = float(args[args.index("-pl") + 1])
                if not (p.min_limit_w <= w <= p.max_limit_w):
                    return 1, "", f"Provided power limit {w} is not a valid power limit\n"
                old = self._limit_w if self._limit_w is not None else p.default_limit_w
                self._limit_w = w
                return 0, f"Power limit set to {w:.2f} W from {old:.2f} W.\n", ""
            if "-lgc" in args:
                lo, hi = [float(x) for x in args[args.index("-lgc") + 1].split(",")]
                self._clock_cap = hi
                return 0, f"GPU clocks set to (gpuClkMin {lo:.0f}, gpuClkMax {hi:.0f}).\n", ""
            if "-rgc" in args:
                self._clock_cap = None
                return 0, "GPU clocks reset.\n", ""
        return 1, "", "unsupported emulated command\n"


# --------------------------------------------------------------------------
def make_backend(kind: str = "auto", index: int = 0, interval_ms: int = 100):
    """kind: auto | nvml | nvidia-smi | fake."""
    if kind == "fake":
        return FakeBackend()
    errors = []
    if kind in ("auto", "nvml"):
        try:
            return NvmlBackend(index)
        except BackendUnavailable as e:
            errors.append(str(e))
            if kind == "nvml":
                raise
    if kind in ("auto", "nvidia-smi"):
        try:
            return NvidiaSmiBackend(index, interval_ms)
        except BackendUnavailable as e:
            errors.append(str(e))
            raise BackendUnavailable("; ".join(errors)) from e
    raise ValueError(f"unknown backend {kind!r}")


# --------------------------------------------------------------------------
def bin_series(t_s: np.ndarray, values: np.ndarray, n_bins: int, step_s: float = 1.0) -> np.ndarray:
    """Mean of ``values`` in consecutive bins [i*step, (i+1)*step), NaN if empty."""
    t = np.asarray(t_s, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    idx = np.floor(t / step_s).astype(np.int64)
    ok = (idx >= 0) & (idx < n_bins) & np.isfinite(v)
    sums = np.bincount(idx[ok], weights=v[ok], minlength=n_bins)[:n_bins]
    cnt = np.bincount(idx[ok], minlength=n_bins)[:n_bins]
    out = np.full(n_bins, np.nan)
    nz = cnt > 0
    out[nz] = sums[nz] / cnt[nz]
    return out


def iso_utc(unix_s: float) -> str:
    return datetime.fromtimestamp(unix_s, tz=timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Sample:
    t_mono_s: float
    t_unix: float
    values: dict = field(default_factory=dict)


class Sampler(GuardedThread):
    """Polls a backend at ``hz`` and writes a CSV, flushing every second."""

    def __init__(self, backend, csv_path: str | Path, stop_event: threading.Event,
                 hz: float = 10.0, flush_every_s: float = 1.0, buffer_s: float = 7200.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__("sampler", stop_event)
        self.backend = backend
        self.csv_path = Path(csv_path)
        self.hz = float(hz)
        self.flush_every_s = flush_every_s
        self._clock = clock
        self._buf: deque[Sample] = deque(maxlen=int(buffer_s * hz) + 10)
        self._lock = threading.Lock()
        self.started = threading.Event()
        self.start_unix = NAN
        self._start_mono = NAN
        self.n_samples = 0
        self.energy_source = "unknown"

    # --- timeline helpers ----------------------------------------------
    def elapsed_s(self) -> float:
        if self._start_mono != self._start_mono:
            return 0.0
        return self._clock() - self._start_mono

    def window(self, last_s: float) -> list[Sample]:
        with self._lock:
            if not self._buf:
                return []
            t_end = self._buf[-1].t_mono_s
            return [s for s in self._buf if s.t_mono_s >= t_end - last_s]

    def latest(self) -> Optional[Sample]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def unix_to_rel(self, unix_s: float) -> float:
        return unix_s - self.start_unix

    # --- thread body -----------------------------------------------------
    def loop(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        kind = getattr(self.backend, "data_kind", "measured")
        period = 1.0 / self.hz
        self._start_mono = self._clock()
        self.start_unix = time.time()
        self.started.set()
        e0: Optional[float] = None
        e_int = 0.0
        prev: Optional[tuple[float, float]] = None
        last_flush = self._clock()
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            i = 0
            while not self.stop_event.is_set():
                t_mono = self._clock() - self._start_mono
                t_unix = self.start_unix + t_mono
                vals = self.backend.read()
                e = vals.get("energy_j", NAN)
                if e == e:
                    if e0 is None:
                        e0 = e
                        self.energy_source = "backend_energy_counter"
                    vals["energy_j"] = e - e0
                else:
                    p_w = vals.get("power_w", NAN)
                    if p_w == p_w and prev is not None:
                        e_int += 0.5 * (p_w + prev[1]) * (t_mono - prev[0])
                    vals["energy_j"] = e_int
                    self.energy_source = "integrated_power"
                p_w = vals.get("power_w", NAN)
                if p_w == p_w:
                    prev = (t_mono, p_w)
                w.writerow([iso_utc(t_unix), f"{t_mono:.3f}"]
                           + [("" if vals[k] != vals[k] else f"{vals[k]:.4f}") for k in SAMPLE_FIELDS]
                           + [kind])
                with self._lock:
                    self._buf.append(Sample(t_mono, t_unix, vals))
                self.n_samples += 1
                if self._clock() - last_flush >= self.flush_every_s:
                    f.flush()
                    last_flush = self._clock()
                i += 1
                nxt = i * period
                wait = nxt - (self._clock() - self._start_mono)
                if wait < -period:  # fell behind; resync instead of bursting
                    i = int((self._clock() - self._start_mono) / period) + 1
                    wait = 0.0
                if wait > 0:
                    self.stop_event.wait(wait)
            f.flush()


def read_csv(path: str | Path):
    """Load a telemetry CSV into a pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(path)
