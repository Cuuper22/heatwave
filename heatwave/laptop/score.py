"""Score persisted forecasts against the logged actuals.

    python -m heatwave.laptop.score laptop/logs/<run>

Reads ``telemetry.csv``, ``forecasts.jsonl`` and ``meta.json`` from the run
directory and writes ``results/laptop/<run>/forecast_score.json`` (through
``heatwave.provenance.write_result``). Only forecasts whose whole horizon lies
inside the logged data are scored. Actuals are the 1 s bin means of the 10 Hz
telemetry, on the same grid the forecasts were issued on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heatwave.forecast import QuantileForecast
from heatwave.forecast.metrics import coverage, mae_median, pinball, wql
from heatwave.provenance import REPO_ROOT, write_result

from .telemetry import bin_series, read_csv

VARIATES = ("power_w", "temp_c")


def load_forecasts(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_run(run_dir: str | Path, out_dir: str | Path | None = None) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    df = read_csv(run_dir / "telemetry.csv")
    fcs = load_forecasts(run_dir / "forecasts.jsonl")
    n_bins = int(np.floor(df["t_mono_s"].max())) + 1 if len(df) else 0
    # The last bin may be partial; do not score against it.
    complete_bins = max(n_bins - 1, 0)
    actual = {v: bin_series(df["t_mono_s"].to_numpy(), df[v].to_numpy(), n_bins) for v in VARIATES}

    usable = [f for f in fcs if f["origin_s"] + f["horizon"] <= complete_bins]
    result: dict = {
        "run": run_dir.name,
        "data_kind": "simulated" if meta.get("data_kind") == "simulated" else "measured",
        "n_forecasts_issued": len(fcs),
        "n_forecasts_scored": len(usable),
        "complete_bins": complete_bins,
        "methods": {},
        "notes": "Forecasts whose full horizon is inside the log are scored; actuals are 1 s bin means.",
    }
    if not usable:
        result["methods"] = {}
        write = _write(result, run_dir, out_dir)
        return write
    horizon = usable[0]["horizon"]
    usable = [f for f in usable if f["horizon"] == horizon]
    methods = [m for m in usable[0]["forecasts"] if all(m in f["forecasts"] for f in usable)]
    for m in methods:
        per_var = {}
        for vi, v in enumerate(VARIATES):
            vals = np.array([f["forecasts"][m][v] for f in usable], dtype=np.float64)  # (N,H,21)
            fc = QuantileForecast(vals[:, None, :, :], method=m)
            act = np.stack([actual[v][f["origin_s"]: f["origin_s"] + horizon] for f in usable])[:, None, :]
            ok = ~np.isnan(act).any(axis=(1, 2))
            if not ok.any():
                continue
            fc = QuantileForecast(vals[ok][:, None, :, :], method=m)
            act = act[ok]
            per_var[v] = {
                "n_forecasts": int(ok.sum()),
                "pinball_mean": pinball(fc, act),
                "wql": wql(fc, act),
                "coverage_80": coverage(fc, act, 0.8),
                "coverage_98": coverage(fc, act, 0.98),
                "mae_median": mae_median(fc, act),
            }
        result["methods"][m] = per_var
    # Paired view: share of forecasts where the first method has the lower mean pinball loss.
    if len(methods) == 2:
        first, second = methods
        wins = {}
        for vi, v in enumerate(VARIATES):
            losses = []
            for m in methods:
                vals = np.array([f["forecasts"][m][v] for f in usable])
                act = np.stack([actual[v][f["origin_s"]: f["origin_s"] + horizon] for f in usable])
                lv = np.asarray(usable[0]["levels"])
                d = act[:, :, None] - vals
                loss = np.maximum(lv * d, (lv - 1) * d).mean(axis=(1, 2))
                losses.append(loss)
            ok = ~(np.isnan(losses[0]) | np.isnan(losses[1]))
            wins[v] = {"method": first, "n_lower_pinball": int((losses[0][ok] < losses[1][ok]).sum()),
                       "n_compared": int(ok.sum())}
        result["paired_lower_pinball"] = wins
    return _write(result, run_dir, out_dir, horizon)


def _write(result: dict, run_dir: Path, out_dir, horizon: int | None = None) -> dict:
    if horizon is not None:
        result["horizon_s"] = horizon
    out = Path(out_dir) if out_dir else REPO_ROOT / "results" / "laptop" / run_dir.name
    rec = write_result(
        out / "forecast_score.json", result, script=__file__,
        data_kind=result["data_kind"],
        inputs=[run_dir / "telemetry.csv", run_dir / "forecasts.jsonl", run_dir / "meta.json"],
        notes=("SIMULATED: produced with the fake backend, not a GPU measurement."
               if result["data_kind"] == "simulated" else "Measured on the laptop GPU."),
    )
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a laptop run's forecasts.")
    ap.add_argument("run_dir")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    rec = score_run(a.run_dir, a.out_dir)
    r = rec["result"]
    print(f"scored {r['n_forecasts_scored']} of {r['n_forecasts_issued']} forecasts ({r['data_kind']})")
    for m, per in r["methods"].items():
        for v, s in per.items():
            print(f"  {m:24s} {v:8s} pinball={s['pinball_mean']:.4f} wql={s['wql']:.4f} "
                  f"cov80={s['coverage_80']:.2f} cov98={s['coverage_98']:.2f} mae={s['mae_median']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
