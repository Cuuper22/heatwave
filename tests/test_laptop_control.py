"""Controller tests, all against the SIMULATED backend's emulated nvidia-smi."""

import math
import signal

import numpy as np
import pytest

from heatwave.forecast import QuantileForecast
from heatwave.laptop.control import (
    ControlConfig, Controller, classify_smi_error, install_exit_handlers,
)
from heatwave.laptop.forecast_loop import ForecastRecord
from heatwave.laptop.run import build_parser
from heatwave.laptop.telemetry import FakeBackend, FakeParams


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(smi_mode="power", admin=True, **cfg_kw):
    backend = FakeBackend(FakeParams(smi_mode=smi_mode))
    lines = []
    cfg = ControlConfig(**cfg_kw)
    clock = Clock()
    ctl = Controller(backend.info(), cfg, runner=backend.smi, is_admin=admin,
                     say=lines.append, clock=clock)
    ctl.setup()
    return ctl, backend, lines, clock


def record(power_q, temp_q, h=10, issued=0.0, name="timestral-5M"):
    vals = np.zeros((1, 2, h, 21))
    vals[0, 0] = power_q
    vals[0, 1] = temp_q
    rec = ForecastRecord(origin_s=0, horizon=h, issued_unix=0.0, issued_mono_s=issued,
                         context_len=100, period_s=None)
    rec.forecasts[name] = QuantileForecast(vals, method=name)
    return rec


# --- defaults ---------------------------------------------------------------
def test_live_actuation_is_the_default():
    assert ControlConfig().dry_run is False
    ns = build_parser().parse_args([])
    assert ns.dry_run is False
    assert build_parser().parse_args(["--dry-run"]).dry_run is True


def test_dry_run_never_calls_nvidia_smi_and_prints_commands():
    ctl, backend, lines, clock = make(dry_run=True)
    n_setup = len(backend.smi_log)
    clock.t = 100
    rec = ctl.set_limit(60, now=clock.t)
    assert rec["status"] == "dry_run"
    assert len(backend.smi_log) == n_setup  # nothing was run
    assert any("DRY RUN, not run: nvidia-smi -i 0 -" in l for l in lines)
    assert ctl.current != ctl.hi or ctl.mode == "power"
    ctl.release()
    assert len(backend.smi_log) == n_setup  # release in dry run runs nothing


def test_live_mode_prints_exact_command_and_runs_it():
    ctl, backend, lines, clock = make("power")
    clock.t = 100
    rec = ctl.set_limit(77, now=clock.t)
    assert rec["status"] == "applied" and rec["command"] == "nvidia-smi -i 0 -pl 77"
    assert backend.smi_log[-1] == ["nvidia-smi", "-i", "0", "-pl", "77"]
    assert any("running: nvidia-smi -i 0 -pl 77" in l for l in lines)


# --- clamping -----------------------------------------------------------------
def test_clamp_to_reported_range_and_floor():
    ctl, backend, _, _ = make("power")
    assert ctl.hi == 105.0
    assert ctl.lo == pytest.approx(0.5 * 80.0)  # floor above the reported min of 1 W
    assert ctl.clamp(500) == 105.0
    assert ctl.clamp(3) == pytest.approx(40.0)


def test_clamp_applies_to_requests_even_with_huge_steps():
    ctl, backend, _, clock = make("power", max_step_w=1000.0, min_interval_s=0.0)
    ctl.set_limit(-50, now=1)
    assert backend._limit_w == pytest.approx(40.0)
    ctl.set_limit(10_000, now=2)
    assert backend._limit_w == pytest.approx(105.0)


def test_floor_never_below_reported_min():
    backend = FakeBackend(FakeParams(smi_mode="power", min_limit_w=60.0))
    ctl = Controller(backend.info(), ControlConfig(), runner=backend.smi, is_admin=True, say=lambda s: None)
    assert ctl.lo == 60.0


# --- rate limiting --------------------------------------------------------------
def test_one_change_per_interval_and_max_step():
    ctl, backend, _, _ = make("power", min_interval_s=10.0, max_step_w=5.0)
    a = ctl.set_limit(60, now=100)
    assert a["status"] == "applied" and a["new"] == 75.0  # 80 -> 75, step limited
    b = ctl.set_limit(60, now=105)
    assert b["status"] == "rate_limited" and backend._limit_w == 75.0
    c = ctl.set_limit(60, now=111)
    assert c["status"] == "applied" and c["new"] == 70.0


def test_small_changes_are_not_sent():
    ctl, backend, _, _ = make("power", min_change_w=2.0)
    n = len(backend.smi_log)
    assert ctl.set_limit(ctl.current - 1, now=100)["status"] == "no_change"
    assert len(backend.smi_log) == n


# --- fallback selection ------------------------------------------------------------
def test_power_limit_used_when_supported():
    ctl, _, _, _ = make("power")
    assert ctl.mode == "power" and not ctl.dry_run


def test_falls_back_to_clock_limit_when_pl_unsupported():
    ctl, backend, lines, clock = make("clock")
    assert ctl.mode == "clock" and not ctl.dry_run and ctl.unit == "MHz"
    assert any("-pl did not work" in l for l in lines)
    ctl.set_limit(1000, now=100)
    assert backend.smi_log[-1][3] == "-lgc" and backend.smi_log[-1][4].endswith(",1600")


def test_stays_dry_without_permission_and_says_why():
    ctl, backend, lines, _ = make("noadmin")
    assert ctl.dry_run
    assert "permission" in ctl.mode_reason
    assert any("DRY RUN" in l for l in lines)


def test_stays_dry_when_not_admin_without_touching_the_gpu():
    ctl, backend, _, _ = make("power", admin=False)
    assert ctl.dry_run and "administrator" in ctl.mode_reason
    assert backend.smi_log == []


def test_failed_command_drops_to_dry_run():
    ctl, backend, _, _ = make("power", min_interval_s=0.0)
    backend.p.smi_mode = "noadmin"
    rec = ctl.set_limit(60, now=1)
    assert rec["status"] == "failed" and ctl.dry_run


def test_classify_smi_error():
    assert classify_smi_error("Insufficient Permissions") == "permission"
    assert classify_smi_error("Access is denied") == "permission"
    assert classify_smi_error("Not Supported") == "unsupported"
    assert classify_smi_error("something odd") == "unknown"


# --- exit: set to the maximum and leave it there ---------------------------------------
def test_normal_exit_sets_power_limit_to_maximum():
    ctl, backend, _, _ = make("power", min_interval_s=0.0)
    with ctl:
        ctl.set_limit(60, now=1)
        assert backend._limit_w < 105.0
    assert backend._limit_w == 105.0
    assert ctl.released
    assert backend.smi_log[-1] == ["nvidia-smi", "-i", "0", "-pl", "105"]


def test_normal_exit_resets_clocks_in_clock_mode():
    ctl, backend, _, _ = make("clock", min_interval_s=0.0)
    with ctl:
        ctl.set_limit(1000, now=1)
        assert backend._clock_cap is not None
    assert backend._clock_cap is None
    assert backend.smi_log[-1] == ["nvidia-smi", "-i", "0", "-rgc"]


def test_exit_on_exception_still_releases_and_reraises():
    ctl, backend, _, _ = make("power", min_interval_s=0.0)
    with pytest.raises(RuntimeError):
        with ctl:
            ctl.set_limit(60, now=1)
            raise RuntimeError("boom")
    assert backend._limit_w == 105.0 and ctl.released


@pytest.mark.parametrize("sig,exc", [(signal.SIGINT, KeyboardInterrupt), (signal.SIGTERM, SystemExit)])
def test_exit_on_signals(sig, exc):
    ctl, backend, _, _ = make("power", min_interval_s=0.0)
    with pytest.raises(exc):
        with ctl:
            ctl.set_limit(60, now=1)
            signal.raise_signal(sig)
    assert backend._limit_w == 105.0 and ctl.released


def test_signal_handlers_are_removed_afterwards():
    ctl, _, _, _ = make("power")
    before = signal.getsignal(signal.SIGTERM)
    uninstall = install_exit_handlers(ctl)
    assert signal.getsignal(signal.SIGTERM) is not before
    uninstall()
    assert signal.getsignal(signal.SIGTERM) is before


def test_release_is_idempotent_and_blocks_later_changes():
    ctl, backend, _, _ = make("power", min_interval_s=0.0)
    ctl.set_limit(60, now=1)
    assert ctl.release() and ctl.release()
    n = len(backend.smi_log)
    assert ctl.set_limit(50, now=5)["status"] == "released"
    assert len(backend.smi_log) == n


def test_release_failure_reports_manual_command_and_can_retry():
    ctl, backend, lines, _ = make("power", min_interval_s=0.0)
    ctl.set_limit(60, now=1)
    backend.p.smi_mode = "noadmin"
    assert ctl.release() is False and ctl.release_failed
    assert any("Run this yourself: nvidia-smi -i 0 -pl 105" in l for l in lines)
    backend.p.smi_mode = "power"
    assert ctl.release() is True


def test_nothing_changed_means_nothing_to_release():
    ctl, backend, _, _ = make("power", admin=False)
    ctl.release()
    assert backend.smi_log == []


# --- decisions from forecasts ----------------------------------------------------------
def test_hot_forecast_lowers_the_limit():
    ctl, backend, _, _ = make("power", temp_limit_c=75.0, quantile=0.9)
    rec = record(60.0, 85.0)
    out = ctl.step(rec, 70.0, 60.0, now_s=1.0, now=100)
    assert out["status"] == "applied" and out["new"] < 80.0
    assert "temp peak" in out["reason"]


def test_cool_forecast_raises_the_limit_but_not_above_max():
    ctl, backend, _, _ = make("power", temp_limit_c=75.0)
    ctl.set_limit(60, now=1)
    cur = ctl.current
    out = ctl.step(record(40.0, 55.0), 55.0, 40.0, now_s=1.0, now=100)
    assert out["new"] > cur
    for k in range(100):
        ctl.step(record(40.0, 55.0), 55.0, 40.0, now_s=1.0, now=200 + 20 * k)
    assert ctl.current == 105.0


def test_forecast_quantile_is_what_the_controller_uses():
    levels = np.linspace(40, 90, 21)  # q50 = 65, q90 = 85, q99 = 89.5
    temp = np.tile(levels, (10, 1))
    rec = record(np.full((10, 21), 50.0), temp)
    ctl, _, _, _ = make("power", temp_limit_c=80.0, quantile=0.5)
    ctl.step(rec, 60.0, 50.0, 1.0, now=100)  # q50 peak 65 C is under 80 C: no reduction
    assert ctl.current >= 80.0
    ctl2, _, _, _ = make("power", temp_limit_c=80.0, quantile=0.9)
    assert ctl2.step(rec, 60.0, 50.0, 1.0, now=100)["status"] == "applied"
    assert ctl2.current < 80.0


def test_power_budget_lowers_the_limit():
    ctl, _, _, _ = make("power", temp_limit_c=None, power_budget_w=50.0)
    out = ctl.step(record(70.0, 50.0), 50.0, 70.0, 1.0, now=100)
    assert out["status"] == "applied" and out["new"] < 80.0 and "power peak" in out["reason"]


def test_stale_or_missing_forecast_holds():
    ctl, _, _, _ = make("power")
    assert ctl.step(None, 50.0, 40.0, 1.0)["status"] == "hold"
    stale = record(60.0, 90.0, issued=0.0)
    assert ctl.step(stale, 50.0, 40.0, now_s=500.0)["status"] == "hold"
    assert ctl.current == 80.0


def test_clock_mode_reduces_clock_on_hot_forecast():
    ctl, backend, _, _ = make("clock", temp_limit_c=75.0)
    out = ctl.step(record(60.0, 85.0), 70.0, 60.0, 1.0, now=100)
    assert out["status"] == "applied" and out["new"] < 1700.0
    assert backend.smi_log[-1][3] == "-lgc"
