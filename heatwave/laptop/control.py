"""Forecast-driven power / clock limit controller for one GPU.

Each cycle the controller reads the latest forecast, takes the configured
quantile q of GPU temperature (and, optionally, of power) over the forecast
horizon, and moves a limit so that the q-quantile peak stays at or under a
temperature setpoint and / or a power budget.

Actuation modes
  power  nvidia-smi -i N -pl <watts>
  clock  nvidia-smi -i N -lgc <min>,<max>   (fallback when -pl is not supported,
         common on laptop GPUs where Dynamic Boost manages power)
  dry    nothing is run; the commands are printed and logged

Defaults and safety
  * Real actuation is the default. ``--dry-run`` (cfg.dry_run) is the opt-in.
  * Without administrator rights, or when neither -pl nor -lgc works, the
    controller stays in dry run and logs why.
  * Every limit is clamped to the GPU's reported [min, max], never below the
    floor (default 50% of the default limit, or of the max SM clock).
  * Changes are rate limited (one per ``min_interval_s``) and step limited.
  * On exit (normal, Ctrl+C, SIGTERM, Windows CTRL_BREAK, or an exception) the
    GPU is set to its maximum allowed power limit (or ``nvidia-smi -rgc`` in
    clock mode) and left there. Nothing is restored to the earlier value.
"""

from __future__ import annotations

import atexit
import math
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .state import GuardedThread, RunState, json_text
from .telemetry import GpuInfo

Runner = Callable[[Sequence[str]], "tuple[int, str, str]"]


@dataclass
class ControlConfig:
    quantile: float = 0.9
    temp_limit_c: Optional[float] = 75.0
    power_budget_w: Optional[float] = None
    floor_frac: float = 0.5
    min_interval_s: float = 10.0
    max_step_w: float = 5.0
    max_step_mhz: float = 100.0
    # Tuning knobs (not measurements): response per degree / watt of forecast excess.
    gain_w_per_c: float = 3.0
    gain_mhz_per_c: float = 60.0
    gain_mhz_per_w: float = 10.0
    raise_w: float = 1.0
    raise_mhz: float = 30.0
    hysteresis_c: float = 2.0
    hysteresis_w: float = 3.0
    min_change_w: float = 1.0
    min_change_mhz: float = 15.0
    max_forecast_age_s: float = 30.0
    dry_run: bool = False
    gpu_index: int = 0
    forecaster: str = "timestral"  # prefix of the forecast method that drives control


def real_runner(args: Sequence[str]) -> tuple[int, str, str]:
    try:
        r = subprocess.run(list(args), capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", "nvidia-smi was not found"
    except subprocess.TimeoutExpired:
        return 124, "", "nvidia-smi timed out"


def detect_admin() -> bool:
    """True when the process can change GPU limits (Windows admin or root)."""
    if os.name == "nt":
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


def classify_smi_error(text: str) -> str:
    """'permission', 'unsupported' or 'unknown'. Keyword match; wording varies by driver."""
    t = text.lower()
    if any(k in t for k in ("insufficient permission", "administrator", "access is denied",
                            "access denied", "permission denied", "must be root", "requires root")):
        return "permission"
    if any(k in t for k in ("not supported", "not available", "unsupported", "n/a")):
        return "unsupported"
    return "unknown"


class Controller:
    def __init__(self, info: GpuInfo, cfg: ControlConfig, runner: Runner = real_runner,
                 is_admin: Optional[bool] = None, log_path: Optional[str | Path] = None,
                 say: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.info = info
        self.cfg = cfg
        self.runner = runner
        self.is_admin = detect_admin() if is_admin is None else is_admin
        self.say = lambda s: say(f"[control] {s}")
        self.clock = clock
        self._log_f = None
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log_f = open(log_path, "w", encoding="utf-8")
        self.dry_run = cfg.dry_run
        self.mode = "power"
        self.mode_reason = ""
        self.unit = "W"
        self.lo = self.hi = 0.0
        self.original: float = float("nan")
        self.current: float = float("nan")
        self.touched = False
        self.released = False
        self.release_failed = False
        self.last_change_t = -math.inf
        self.n_changes = 0
        self.n_decisions = 0
        self.last: dict = {"status": "not started", "reason": ""}
        self.last_command: Optional[str] = None
        self._lock = threading.RLock()
        self._set_mode_specs("power" if not math.isnan(info.power_limit_w) else "clock")

    # --- mode specs -------------------------------------------------------
    def _set_mode_specs(self, mode: str) -> None:
        i, c = self.info, self.cfg
        self.mode = mode
        if mode == "power":
            self.unit = "W"
            dflt = i.default_limit_w if not math.isnan(i.default_limit_w) else i.max_limit_w
            lo_min = i.min_limit_w if not math.isnan(i.min_limit_w) else 0.0
            self.hi = i.max_limit_w
            self.lo = min(max(lo_min, c.floor_frac * dflt), self.hi)
            self.original = i.power_limit_w if not math.isnan(i.power_limit_w) else dflt
            self.current = min(max(self.original, self.lo), self.hi)
        else:
            self.unit = "MHz"
            self.hi = i.clock_max_mhz
            cmin = i.clock_min_mhz if not math.isnan(i.clock_min_mhz) else 0.0
            self.lo = min(max(cmin, c.floor_frac * self.hi), self.hi) if not math.isnan(self.hi) else float("nan")
            self.original = self.hi  # no cap
            self.current = self.hi

    # --- commands -----------------------------------------------------------
    def _smi(self, *args: str) -> list[str]:
        return ["nvidia-smi", "-i", str(self.cfg.gpu_index), *args]

    def command_for(self, value: float) -> list[str]:
        if self.mode == "power":
            return self._smi("-pl", str(int(round(value))))
        cmin = self.info.clock_min_mhz
        cmin = int(cmin) if not math.isnan(cmin) else int(self.lo)
        return self._smi("-lgc", f"{cmin},{int(round(value))}")

    def release_command(self) -> list[str]:
        if self.mode == "power":
            return self._smi("-pl", str(int(round(self.info.max_limit_w))))
        return self._smi("-rgc")

    def _go_dry(self, reason: str) -> None:
        self.dry_run = True
        self.mode_reason = reason
        self.say(f"DRY RUN: {reason}")

    def setup(self) -> None:
        """Choose the actuation mode (checks that a limit command works)."""
        i, c = self.info, self.cfg
        if c.dry_run:
            self.mode_reason = ("dry run requested (--dry-run); would use "
                                + ("a power limit" if self.mode == "power"
                                   else "an SM clock limit because power.limit reports N/A"))
            self.say(f"DRY RUN: {self.mode_reason}")
            return
        if math.isnan(i.max_limit_w) and math.isnan(i.clock_max_mhz):
            self._go_dry("the GPU's limits could not be read")
            return
        if not self.is_admin:
            self._go_dry("no administrator rights; nvidia-smi -pl and -lgc need them "
                         "(start an admin PowerShell, or run as root)")
            return
        self._set_mode_specs("power")
        if not math.isnan(self.hi):
            cmd = self.command_for(self.current)
            self.say(f"checking power-limit support: {' '.join(cmd)}")
            self.touched = True
            rc, out, err = self.runner(cmd)
            if rc == 0:
                self.mode_reason = "nvidia-smi -pl works; using the power limit"
                self.say(self.mode_reason)
                return
            kind = classify_smi_error(out + err)
            if kind == "permission":
                self._go_dry(f"-pl was refused for lack of permission ({err.strip() or out.strip()})")
                return
            self.say(f"-pl did not work ({kind}: {(err or out).strip()}); trying an SM clock limit")
        self._set_mode_specs("clock")
        if math.isnan(self.hi):
            self._go_dry("-pl is unsupported and the max SM clock is unknown, so no fallback is possible")
            return
        cmd = self.command_for(self.current)
        self.say(f"checking clock-limit support: {' '.join(cmd)}")
        self.touched = True
        rc, out, err = self.runner(cmd)
        if rc == 0:
            self.mode_reason = "-pl unsupported; falling back to the SM clock limit (-lgc)"
            self.say(self.mode_reason)
            return
        self._go_dry(f"neither -pl nor -lgc worked ({classify_smi_error(out + err)}: {(err or out).strip()})")

    # --- limits ---------------------------------------------------------------
    def clamp(self, value: float) -> float:
        """Clamp to [floor, max reported]; the floor is never below the reported min."""
        return float(min(max(value, self.lo), self.hi))

    def set_limit(self, value: float, now: Optional[float] = None) -> dict:
        """Request a limit. Clamps, step-limits, rate-limits, then runs or prints the command."""
        with self._lock:
            now = self.clock() if now is None else now
            c = self.cfg
            max_step = c.max_step_w if self.mode == "power" else c.max_step_mhz
            min_change = c.min_change_w if self.mode == "power" else c.min_change_mhz
            clamped = self.clamp(value)
            step = min(max(clamped - self.current, -max_step), max_step)
            new = self.clamp(round(self.current + step))
            rec = {"requested": float(value), "clamped": clamped, "new": new,
                   "previous": self.current, "unit": self.unit, "command": None,
                   "applied": False, "status": "", "rc": None}
            if self.released:
                rec["status"] = "released"  # the GPU was already set to its maximum on exit
                return rec
            if abs(new - self.current) < min_change:
                rec["status"] = "no_change"
                return rec
            if now - self.last_change_t < c.min_interval_s:
                rec["status"] = "rate_limited"
                return rec
            cmd = self.command_for(new)
            text = " ".join(cmd)
            rec["command"] = text
            self.last_command = text
            self.last_change_t = now
            if self.dry_run:
                self.say(f"DRY RUN, not run: {text}")
                rec["status"] = "dry_run"
                self.current = new
                self.n_changes += 1
                return rec
            self.say(f"running: {text}")
            self.touched = True
            rc, out, err = self.runner(cmd)
            rec["rc"] = rc
            if rc == 0:
                rec["status"] = "applied"
                rec["applied"] = True
                self.current = new
                self.n_changes += 1
            else:
                rec["status"] = "failed"
                rec["error"] = (err or out).strip()
                self._go_dry(f"command failed ({rec['error']}); no further changes will be sent")
            return rec

    # --- decision ---------------------------------------------------------------
    def step(self, forecast, measured_temp: float, measured_power: float,
             now_s: float, now: Optional[float] = None) -> dict:
        """One control cycle from the latest forecast record. Returns a log record."""
        c = self.cfg
        with self._lock:
            self.n_decisions += 1
            base = {"t_s": round(now_s, 2), "mode": self.mode, "dry_run": self.dry_run,
                    "current": self.current, "unit": self.unit}
            if forecast is None:
                self.last = {**base, "status": "hold", "reason": "no forecast yet"}
                return self._log(self.last)
            age = now_s - forecast.issued_mono_s
            if age > c.max_forecast_age_s:
                self.last = {**base, "status": "hold", "reason": f"forecast is {age:.0f} s old"}
                return self._log(self.last)
            names = [k for k in forecast.forecasts if k.startswith(c.forecaster)]
            if not names:
                self.last = {**base, "status": "hold", "reason": f"no '{c.forecaster}' forecast"}
                return self._log(self.last)
            fc = forecast.forecasts[names[0]]
            q = fc.quantile(c.quantile)  # (1, V, H)
            peak_p = float(np.nanmax(q[0, 0]))
            peak_t = float(np.nanmax(q[0, 1]))
            if not math.isnan(measured_temp):
                peak_t = max(peak_t, measured_temp)
            gain_c = c.gain_w_per_c if self.mode == "power" else c.gain_mhz_per_c
            gain_w = 1.0 if self.mode == "power" else c.gain_mhz_per_w
            raise_by = c.raise_w if self.mode == "power" else c.raise_mhz
            down, reasons, ok_to_raise = 0.0, [], True
            if c.temp_limit_c is not None:
                ex = peak_t - c.temp_limit_c
                if ex > 0:
                    down = max(down, gain_c * ex)
                    reasons.append(f"q{c.quantile:g} temp peak {peak_t:.1f} C > setpoint {c.temp_limit_c:g} C")
                elif ex > -c.hysteresis_c:
                    ok_to_raise = False
            if c.power_budget_w is not None:
                ex = peak_p - c.power_budget_w
                if ex > 0:
                    down = max(down, gain_w * ex)
                    reasons.append(f"q{c.quantile:g} power peak {peak_p:.1f} W > budget {c.power_budget_w:g} W")
                elif ex > -c.hysteresis_w:
                    ok_to_raise = False
            if c.temp_limit_c is None and c.power_budget_w is None:
                cand, why = self.current, "no setpoint or budget configured"
            elif down > 0:
                cand, why = self.current - down, "; ".join(reasons)
            elif ok_to_raise:
                cand, why = self.current + raise_by, "forecast peaks are inside the margins; raising"
            else:
                cand, why = self.current, "inside the hysteresis band; holding"
            rec = self.set_limit(cand, now)
            self.last = {**base, **{k: rec[k] for k in ("requested", "clamped", "new", "status", "command")},
                         "reason": why, "peak_temp_c": peak_t, "peak_power_w": peak_p,
                         "measured_temp_c": measured_temp, "measured_power_w": measured_power,
                         "forecaster": names[0]}
            return self._log(self.last)

    def _log(self, rec: dict) -> dict:
        if self._log_f is not None:
            self._log_f.write(json_text(rec) + "\n")
            self._log_f.flush()
        return rec

    # --- release -------------------------------------------------------------------
    def release(self) -> bool:
        """Set the GPU to its maximum allowed power / clocks and leave it there.

        Safe to call repeatedly (atexit, signal handler and finally may all
        call it). Returns True when nothing more needs doing.
        """
        with self._lock:
            if self.released:
                return True
            cmd = self.release_command()
            text = " ".join(cmd)
            if self.dry_run or not self.touched:
                if self.dry_run:
                    self.say(f"DRY RUN: nothing was changed; on exit this would run: {text}")
                self.released = True
                return True
            self.say(f"exit: setting the GPU to its maximum: {text}")
            rc, out, err = self.runner(cmd)
            if rc == 0:
                self.released = True
                self.release_failed = False
                self.current = self.hi if self.mode == "clock" else self.info.max_limit_w
                self.say("GPU left at its maximum allowed " + ("power limit" if self.mode == "power" else "clocks"))
                return True
            self.release_failed = True
            self.say(f"WARNING: could not set the GPU to its maximum ({(err or out).strip()}). "
                     f"Run this yourself: {text}")
            return False

    def status(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode, "dry_run": self.dry_run, "mode_reason": self.mode_reason,
                "unit": self.unit, "current": self.current, "original": self.original,
                "lo": self.lo, "hi": self.hi, "last": dict(self.last),
                "last_command": self.last_command, "n_changes": self.n_changes,
                "released": self.released, "release_failed": self.release_failed,
                "quantile": self.cfg.quantile, "temp_limit_c": self.cfg.temp_limit_c,
                "power_budget_w": self.cfg.power_budget_w,
            }

    def close(self) -> None:
        if self._log_f is not None:
            self._log_f.close()
            self._log_f = None

    # --- exit handling ----------------------------------------------------------------
    def __enter__(self) -> "Controller":
        self._uninstall = install_exit_handlers(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.release()
        finally:
            self._uninstall()
        return False  # never swallow the exception


def install_exit_handlers(controller: Controller) -> Callable[[], None]:
    """atexit + SIGINT / SIGTERM / SIGBREAK handlers that release the GPU.

    Signal handlers can only be installed from the main thread; elsewhere only
    atexit is registered. Returns a function that undoes the installation.
    """
    atexit.register(controller.release)
    previous = {}

    def handler(signum, frame):
        controller.release()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    if threading.current_thread() is threading.main_thread():
        names = ["SIGINT", "SIGTERM", "SIGBREAK"]  # SIGBREAK = Windows CTRL_BREAK
        for name in names:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                previous[sig] = signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def uninstall() -> None:
        atexit.unregister(controller.release)
        for sig, old in previous.items():
            try:
                signal.signal(sig, old)
            except (ValueError, OSError):
                pass

    return uninstall


class ControlLoop(GuardedThread):
    """Runs ``Controller.step`` every ``interval_s`` seconds."""

    def __init__(self, controller: Controller, sampler, state: RunState,
                 stop_event: threading.Event, interval_s: float = 5.0) -> None:
        super().__init__("control", stop_event)
        self.c = controller
        self.sampler = sampler
        self.state = state
        self.interval_s = interval_s

    def loop(self) -> None:
        while not self.stop_event.is_set():
            s = self.sampler.latest()
            t = s.values.get("temp_c", float("nan")) if s else float("nan")
            p = s.values.get("power_w", float("nan")) if s else float("nan")
            self.c.step(self.state.get_forecast(), t, p, self.sampler.elapsed_s())
            self.state.set_control(self.c.status())
            self.stop_event.wait(self.interval_s)
        self.state.set_control(self.c.status())
