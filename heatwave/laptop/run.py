"""One command for the whole laptop session.

    python -m heatwave.laptop.run --minutes 20 [--dry-run] [--quantile 0.9]
        [--temp-limit 75] [--model-size 5M] [--no-load]
    python -m heatwave.laptop.run --log-only --minutes 60     # data collection
    python -m heatwave.laptop.run --fake --minutes 1          # SIMULATED, no GPU needed

Starts the telemetry logger, the training load, the forecast loop, the
controller and the dashboard; writes everything into
``laptop/logs/<UTC timestamp>/``; prints a summary at the end.

Real actuation is the default for the controller. ``--dry-run`` only prints
what it would run. On exit (normal, Ctrl+C or an error) the GPU is set to its
maximum allowed power limit (or ``nvidia-smi -rgc`` in clock mode) and left
there.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from heatwave.provenance import REPO_ROOT, write_result

from .state import RunState, json_text


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m heatwave.laptop.run",
                                 description="GPU logger + load + forecast + control + dashboard.")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--log-only", action="store_true", help="logger + load only, no forecast / control / dashboard")
    ap.add_argument("--dry-run", action="store_true",
                    help="controller prints the nvidia-smi commands but never runs them (default: real actuation)")
    ap.add_argument("--apply", action="store_true", help=argparse.SUPPRESS)  # real actuation is the default
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--temp-limit", type=float, default=75.0, help="temperature setpoint in C")
    ap.add_argument("--power-budget", type=float, default=None, help="optional power budget in W")
    ap.add_argument("--model-size", default="5M", choices=["5M", "28M", "123M", "209M"], help="Timestral size")
    ap.add_argument("--no-load", action="store_true", help="do not start the training load")
    ap.add_argument("--load-size", default="small", choices=["tiny", "small", "medium"])
    ap.add_argument("--ckpt-period-s", type=float, default=30.0)
    ap.add_argument("--horizon", type=int, default=60, help="forecast horizon in seconds")
    ap.add_argument("--forecast-every", type=float, default=5.0)
    ap.add_argument("--context", type=int, default=512, help="context length in seconds")
    ap.add_argument("--min-context", type=int, default=20, help="seconds of data before the first forecast")
    ap.add_argument("--control-every", type=float, default=5.0)
    ap.add_argument("--min-interval", type=float, default=10.0, help="minimum seconds between limit changes")
    ap.add_argument("--floor-frac", type=float, default=0.5)
    ap.add_argument("--control-forecaster", default="timestral", choices=["timestral", "seasonal-naive"])
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for Timestral")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--fake", action="store_true", help="SIMULATED backend and load (no GPU needed)")
    ap.add_argument("--fake-smi", default="clock", choices=["clock", "power", "noadmin"],
                    help="which nvidia-smi behavior the simulated GPU emulates")
    return ap


def _stop_load_process(proc: subprocess.Popen, stop_file: Path) -> None:
    stop_file.write_text("stop\n")
    try:
        proc.wait(timeout=45)
        return
    except subprocess.TimeoutExpired:
        pass
    sig = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
    try:
        proc.send_signal(sig)
        proc.wait(timeout=20)
    except Exception:
        print(f"[run] the load process (pid {proc.pid}) did not stop; end it manually", flush=True)


def summarize(log_dir: Path, meta: dict, extra: dict) -> dict:
    import pandas as pd

    df = pd.read_csv(log_dir / "telemetry.csv")
    out = dict(extra)
    if len(df):
        out.update({
            "samples": int(len(df)),
            "duration_s": float(df["t_mono_s"].iloc[-1]),
            "power_w_mean": float(df["power_w"].mean()), "power_w_max": float(df["power_w"].max()),
            "temp_c_mean": float(df["temp_c"].mean()), "temp_c_max": float(df["temp_c"].max()),
            "sm_clock_mhz_mean": float(df["sm_clock_mhz"].mean()),
            "energy_j": float(df["energy_j"].iloc[-1]),
        })
    fpath = log_dir / "forecasts.jsonl"
    out["forecasts_issued"] = sum(1 for _ in open(fpath, encoding="utf-8")) if fpath.exists() else 0
    return out


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from . import telemetry as tm
    from .control import ControlConfig, ControlLoop, Controller, detect_admin, real_runner
    from .dashboard import DashboardServer
    from .forecast_loop import ForecastLoop, ScheduleTracker
    from .load import FakeLoad

    stop = threading.Event()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = stamp + ("-simulated" if a.fake else "")
    log_dir = Path(a.log_dir) if a.log_dir else REPO_ROOT / "laptop" / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    if a.fake:
        backend = tm.FakeBackend(tm.FakeParams(smi_mode=a.fake_smi))
    else:
        backend = tm.make_backend("auto", a.gpu_index)
    info = backend.info()
    data_kind = backend.data_kind
    meta = {
        "run": log_dir.name, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_kind": data_kind, "backend": backend.name, "gpu": info.__dict__,
        "args": vars(a), "python": sys.version.split()[0], "platform": platform.platform(),
        "log_only": a.log_only,
    }
    state = RunState(meta)
    events_path = log_dir / "events.csv"
    sampler = tm.Sampler(backend, log_dir / "telemetry.csv", stop, hz=a.hz)
    threads = []
    load_proc = None
    load_log = None
    stop_file = log_dir / "stop_load.flag"
    controller = None
    dash = None
    summary_extra: dict = {}
    interrupted = False

    if not a.log_only:
        cfg = ControlConfig(
            quantile=a.quantile, temp_limit_c=a.temp_limit, power_budget_w=a.power_budget,
            floor_frac=a.floor_frac, min_interval_s=a.min_interval, dry_run=a.dry_run,
            gpu_index=a.gpu_index, forecaster=a.control_forecaster)
        controller = Controller(
            info, cfg,
            runner=backend.smi if a.fake else real_runner,
            is_admin=True if a.fake else detect_admin(),
            log_path=log_dir / "control.jsonl")

    print(f"[run] {log_dir}", flush=True)
    print(f"[run] GPU: {info.name} (backend {backend.name}, data_kind={data_kind})", flush=True)
    if a.fake:
        print("[run] SIMULATED run: fake backend and fake load, no GPU is read or changed", flush=True)

    guard = controller if controller is not None else contextlib.nullcontext()
    try:
        with guard:
            if controller is not None:
                controller.setup()
                meta["controller"] = controller.status()
                meta["controller"].pop("last", None)
            (log_dir / "meta.json").write_text(json_text(meta, indent=1) + "\n", encoding="utf-8")
            sampler.start()
            sampler.started.wait(5)

            if not a.no_load:
                if a.fake:
                    fl = FakeLoad(backend, events_path, stop)
                    fl.start()
                    threads.append(fl)
                else:
                    load_log = open(log_dir / "load.log", "w", encoding="utf-8")
                    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
                    load_proc = subprocess.Popen(
                        [sys.executable, "-m", "heatwave.laptop.load",
                         "--minutes", str(a.minutes + 1.0), "--events", str(events_path),
                         "--size", a.load_size, "--ckpt-period-s", str(a.ckpt_period_s),
                         "--stop-file", str(stop_file), "--summary", str(log_dir / "load_summary.json")],
                        stdout=load_log, stderr=subprocess.STDOUT, **kw)

            if not a.log_only:
                tracker = ScheduleTracker(events_path, sampler.start_unix)
                state.tracker = tracker
                fl_thread = ForecastLoop(
                    sampler, tracker, state, log_dir / "forecasts.jsonl", stop,
                    model_size=a.model_size, threads=a.threads, interval_s=a.forecast_every,
                    horizon_s=a.horizon, context_s=a.context, min_context_s=a.min_context)
                fl_thread.start()
                threads.append(fl_thread)
                ctl_thread = ControlLoop(controller, sampler, state, stop, interval_s=a.control_every)
                ctl_thread.start()
                threads.append(ctl_thread)
                state.set_control(controller.status())
                if not a.no_dashboard:
                    try:
                        dash = DashboardServer(state, sampler, a.port)
                        dash.start()
                        print(f"[run] dashboard: http://127.0.0.1:{dash.port}/", flush=True)
                    except OSError as e:
                        print(f"[run] dashboard not started ({e}); the run continues", flush=True)
                banner = "DRY RUN" if controller.dry_run else "LIVE CONTROL"
                print(f"[run] controller: {banner}, mode {controller.mode}; {controller.mode_reason}", flush=True)
            print(f"[run] running for {a.minutes:g} minutes; Ctrl+C stops early", flush=True)

            end = time.monotonic() + a.minutes * 60.0
            try:
                while time.monotonic() < end and not stop.is_set():
                    stop.wait(0.5)
                    if load_proc is not None and load_proc.poll() not in (None, 0):
                        print(f"[run] the load process exited with code {load_proc.returncode}; "
                              f"see {log_dir / 'load.log'}", flush=True)
                        break
            except KeyboardInterrupt:
                interrupted = True
                print("\n[run] Ctrl+C: stopping", flush=True)
            finally:
                stop.set()
                for t in threads + [sampler]:
                    t.join(timeout=30)
                if load_proc is not None:
                    _stop_load_process(load_proc, stop_file)
                if dash is not None:
                    dash.stop()
                if load_log is not None:
                    load_log.close()
        # leaving the `with` block released the GPU
    finally:
        backend.close()
        if controller is not None:
            controller.close()

    errors = {t.name: t.error for t in threads + [sampler] if getattr(t, "error", None)}
    summary_extra = {
        "run": log_dir.name, "data_kind": data_kind, "interrupted": interrupted,
        "thread_errors": errors, "energy_source": sampler.energy_source,
    }
    if controller is not None:
        st = controller.status()
        st.pop("last", None)
        summary_extra["controller"] = st
    summ = summarize(log_dir, meta, summary_extra)
    write_result(log_dir / "summary.json", summ, script=__file__, data_kind=data_kind,
                 inputs=[log_dir / "telemetry.csv"],
                 notes="SIMULATED run (fake backend)" if a.fake else "measured on the local GPU")
    print("[run] summary" + (" (SIMULATED)" if a.fake else ""), flush=True)
    print(json_text(summ, indent=1), flush=True)

    if not a.log_only and not a.no_score and (log_dir / "forecasts.jsonl").exists():
        from .score import score_run

        rec = score_run(log_dir)
        r = rec["result"]
        print(f"[run] scored {r['n_forecasts_scored']} of {r['n_forecasts_issued']} forecasts -> "
              f"results/laptop/{log_dir.name}/forecast_score.json", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
