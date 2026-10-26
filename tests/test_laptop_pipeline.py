"""Forecast loop, scoring, dashboard, load and the one-command run (all SIMULATED where a GPU would be)."""

import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from heatwave.forecast import QUANTILE_LEVELS, SeasonalNaiveForecaster
from heatwave.laptop.dashboard import DashboardServer, build_api_state
from heatwave.laptop.forecast_loop import ForecastLoop, ScheduleTracker
from heatwave.laptop.score import score_run
from heatwave.laptop.state import RunState, json_text
from heatwave.laptop.telemetry import FakeBackend, Sampler


def write_events(path, start_unix, ckpt_starts, dur=2.0, every=40):
    lines = ["t_utc,t_unix,event,step,value", f"x,{start_unix:.3f},config,0,{every}"]
    for i, s in enumerate(ckpt_starts):
        lines.append(f"x,{start_unix + s:.3f},ckpt_start,{i},0")
        lines.append(f"x,{start_unix + s + dur:.3f},ckpt_end,{i},0")
    path.write_text("\n".join(lines) + "\n")


def test_schedule_tracker_covariates(tmp_path):
    ev = tmp_path / "events.csv"
    write_events(ev, 1000.0, [10, 20, 30])
    tr = ScheduleTracker(ev, 1000.0)
    tr.refresh()
    assert tr.period() == pytest.approx(10.0)
    assert tr.duration() == pytest.approx(2.0)
    cov = tr.covariates(first_bin=20, origin=36, horizon=20)  # bins 20..55
    assert cov.shape == (2, 36)
    active = cov[0]
    assert active[0] == pytest.approx(1.0)       # bin 20 is inside [20, 22)
    assert active[5] == 0.0                      # bin 25
    assert active[40 - 20] == pytest.approx(1.0)  # bin 40: predicted [40, 42)
    assert cov[1][30 - 20] == pytest.approx(0.05)  # bin 30 centre is 0.5 s into a 10 s cycle
    assert cov[1].max() <= 1.5


def test_tracker_ignores_partial_last_line(tmp_path):
    ev = tmp_path / "e.csv"
    ev.write_text("t_utc,t_unix,event,step,value\nx,1001.0,ckpt_start,1,0\nx,1002")
    tr = ScheduleTracker(ev, 1000.0)
    tr.refresh()
    assert tr.starts == [1.0] and tr.ends == []
    with open(ev, "a") as f:
        f.write(".0,ckpt_end,1,0\n")
    tr.refresh()
    assert tr.ends == [2.0]


def test_forecast_loop_with_timestral_and_baseline(tmp_path):
    pytest.importorskip("timestral")
    stop = threading.Event()
    backend = FakeBackend()
    sampler = Sampler(backend, tmp_path / "t.csv", stop, hz=10.0)
    sampler.start()
    sampler.started.wait(2)
    write_events(tmp_path / "events.csv", sampler.start_unix, [0.5, 1.5, 2.5], dur=0.4)
    tracker = ScheduleTracker(tmp_path / "events.csv", sampler.start_unix)
    state = RunState({"data_kind": "simulated"})
    state.tracker = tracker
    loop = ForecastLoop(sampler, tracker, state, tmp_path / "f.jsonl", stop,
                        model_size="5M", horizon_s=8, min_context_s=3)
    backend.set_activity(1.0)
    time.sleep(4.2)
    rec = loop.forecast_once()
    stop.set()
    sampler.join(5)
    assert rec is not None and rec.horizon == 8
    names = list(rec.forecasts)
    assert names[0].startswith("timestral") and names[1] == "seasonal-naive"
    for fc in rec.forecasts.values():
        assert fc.values.shape == (1, 2, 8, 21)
        assert np.all(np.isfinite(fc.values))
    row = json.loads(rec.to_json())
    assert row["levels"] == list(QUANTILE_LEVELS) and row["variates"] == ["power_w", "temp_c"]
    assert len(row["forecasts"][names[0]]["power_w"]) == 8
    # the dashboard content works with this record
    state.set_forecast(rec)
    api = build_api_state(state, sampler)
    assert api["forecast"]["horizon"] == 8 and api["history"]["power_w"]
    json.loads(json_text(api))


def _synthetic_run(tmp_path, perfect=True):
    """A run directory whose forecasts are exact (or deliberately offset)."""
    run = tmp_path / "run"
    run.mkdir()
    n = 80
    t = np.arange(0, n, 0.1)
    power = 50 + 10 * np.sin(t / 5)
    temp = 60 + 0.1 * t
    lines = ["t_utc,t_mono_s,power_w,energy_j,temp_c,sm_clock_mhz,mem_clock_mhz,util_gpu_pct,"
             "util_mem_pct,fan_pct,power_limit_w,enforced_limit_w,mem_used_mib,data_kind"]
    for i, ti in enumerate(t):
        lines.append(f"x,{ti:.3f},{power[i]:.4f},0,{temp[i]:.4f},1000,7000,100,10,,,80,1000,simulated")
    (run / "telemetry.csv").write_text("\n".join(lines) + "\n")
    (run / "meta.json").write_text(json_text({"data_kind": "simulated"}))
    H = 5
    rows = []
    for origin in (20, 30, 40, 77):  # the last one runs past the log and must be skipped
        def block(series, off):
            bins = np.array([series[(t >= origin + j) & (t < origin + j + 1)].mean()
                             if origin + j < n else np.nan for j in range(H)]) + off
            return (np.nan_to_num(bins)[:, None] + np.linspace(-0.01, 0.01, 21)[None, :]).tolist()
        rows.append({"origin_s": origin, "horizon": H, "levels": list(QUANTILE_LEVELS),
                     "forecasts": {"exact": {"power_w": block(power, 0.0), "temp_c": block(temp, 0.0)},
                                   "offset": {"power_w": block(power, 5.0), "temp_c": block(temp, 5.0)}}})
    (run / "forecasts.jsonl").write_text("\n".join(json_text(r) for r in rows) + "\n")
    return run


def test_score_run_exact_and_offset_forecasts(tmp_path):
    run = _synthetic_run(tmp_path)
    rec = score_run(run, out_dir=tmp_path / "out")
    r = rec["result"]
    assert r["n_forecasts_issued"] == 4 and r["n_forecasts_scored"] == 3
    assert rec["provenance"]["data_kind"] == "simulated"
    ex = r["methods"]["exact"]["power_w"]
    assert ex["pinball_mean"] == pytest.approx(0.0, abs=1e-3)
    assert ex["mae_median"] == pytest.approx(0.0, abs=1e-3)
    assert ex["coverage_80"] == 1.0 and ex["coverage_98"] == 1.0
    off = r["methods"]["offset"]["power_w"]
    assert off["mae_median"] == pytest.approx(5.0, abs=1e-3)
    assert off["coverage_80"] == 0.0
    assert (tmp_path / "out" / "forecast_score.json").exists()


def test_dashboard_http(tmp_path):
    stop = threading.Event()
    sampler = Sampler(FakeBackend(), tmp_path / "t.csv", stop, hz=20.0)
    sampler.start()
    sampler.started.wait(2)
    time.sleep(0.5)
    state = RunState({"data_kind": "simulated"})
    state.set_control({"dry_run": True, "mode": "clock", "unit": "MHz", "current": 1700.0})
    srv = DashboardServer(state, sampler, port=0)
    srv.start()
    try:
        base = f"http://127.0.0.1:{srv.port}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "DRY RUN" in page and "LIVE CONTROL" in page and "SIMULATED DATA" in page
        assert "http://" not in page.replace("http://127.0.0.1", "")  # nothing loaded from elsewhere
        api = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert api["control"]["dry_run"] is True and api["meta"]["data_kind"] == "simulated"
        assert api["latest"]["power_w"] is not None
    finally:
        srv.stop()
        stop.set()
        sampler.join(5)


def test_load_on_cpu_writes_step_and_checkpoint_events(tmp_path):
    pytest.importorskip("torch")
    from heatwave.laptop.load import LoadConfig, run_load

    out = run_load(LoadConfig(size="nano", minutes=0.15, ckpt_every=4, device="cpu", warmup_steps=2),
                   tmp_path / "events.csv")
    text = (tmp_path / "events.csv").read_text().splitlines()
    kinds = [l.split(",")[2] for l in text[1:]]
    assert kinds[0] == "config" and kinds[-1] == "end"
    assert kinds.count("step") >= 4 and kinds.count("ckpt_start") == kinds.count("ckpt_end") >= 1
    assert out["checkpoints"] == kinds.count("ckpt_end")


def test_one_command_fake_run_end_to_end(tmp_path):
    pytest.importorskip("timestral")
    from heatwave.laptop.run import main

    log = tmp_path / "log"
    rc = main(["--fake", "--minutes", "0.45", "--horizon", "6", "--forecast-every", "2",
               "--min-context", "6", "--control-every", "2", "--min-interval", "4",
               "--temp-limit", "50", "--port", "0", "--no-score", "--log-dir", str(log)])
    assert rc == 0
    for name in ("telemetry.csv", "events.csv", "forecasts.jsonl", "control.jsonl", "meta.json", "summary.json"):
        assert (log / name).exists(), name
    meta = json.loads((log / "meta.json").read_text())
    assert meta["data_kind"] == "simulated"
    summ = json.loads((log / "summary.json").read_text())
    assert summ["provenance"]["data_kind"] == "simulated"
    assert summ["result"]["controller"]["dry_run"] is False  # live is the default
    assert summ["result"]["controller"]["released"] is True
    assert summ["result"]["controller"]["n_changes"] >= 1
    rec = score_run(log, out_dir=tmp_path / "scores")
    assert rec["result"]["n_forecasts_issued"] >= 1


def test_log_only_run_has_no_forecast_or_control(tmp_path):
    from heatwave.laptop.run import main

    log = tmp_path / "log"
    rc = main(["--fake", "--log-only", "--minutes", "0.1", "--log-dir", str(log)])
    assert rc == 0
    assert (log / "telemetry.csv").exists() and (log / "events.csv").exists()
    assert not (log / "forecasts.jsonl").exists() and not (log / "control.jsonl").exists()
