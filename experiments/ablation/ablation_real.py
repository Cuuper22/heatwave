"""Covariate ablation on REAL data: public grid, carbon and weather series, and the MEASURED laptop GPU runs.

Datasets (each labelled public or measured):
  caiso_rt_lmp_5min   public    CAISO NP15 real-time LMP; covariates DA (day-ahead price), WX (Bay Area weather forecast,
                                day-ahead vintage), DEM (CAISO day-ahead demand forecast)
  caiso_carbon_5min   public    CAISO in-state carbon intensity; covariates WX, DEM
  gb_carbon_30min     public    GB carbon intensity (actual); covariates F (archived GB intensity forecast, vintage
                                undocumented, assumed 24 h lead), WX (London weather forecast, day-ahead vintage)
  laptop_*            measured  RTX 3060 Laptop GPU run2 (test run) and run1 (calibration run, zero-shot model):
                                power (energy-counter) and temperature at 1 s, power at 10 Hz; covariates S (scheduled
                                load-active indicator), K (checkpoint indicator projected from the schedule and the last
                                observed checkpoint), L_realised (logged checkpoint intervals of the whole run: they hold
                                the realised start and duration, so they are not known at the origin; kept as a labelled
                                comparison).
Grid and carbon series are scored on origins from 2026-05-01 (fixed test split). Zero-shot Timestral (or a fine-tuned
checkpoint via --weights-dir) and a seasonal-naive reference; no parameter is fitted on any series.

Usage (development): python experiments/ablation/ablation_real.py --sizes 5M --stride-mult 8 --reps 200 --out results/ablation/dev/real.json
Usage (Colab A100):  python experiments/ablation/ablation_real.py --sizes 209M 28M --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import abl_common as ac
import abl_data as ad
from abl_engine import MethodSpec, evaluate, result_block
from heatwave.forecast import SeasonalNaiveForecaster
from heatwave.forecast import covariates as cv
from heatwave.provenance import write_result

LAPTOP_SETS = [("power", "1s"), ("temp", "1s"), ("power", "10hz")]


def gb_vintage_diagnostic() -> dict:
    """Error of the archived GB intensity forecast against the actual, next to persistence at several leads (test period)."""
    from heatwave.data import public as P

    a = P.load_carbon("gb", "actual")["intensity_g_per_kwh"]
    f = P.load_carbon("gb", "forecast")["intensity_g_per_kwh"]
    x = pd.concat([a, f], axis=1, keys=["a", "f"])[ad.TEST_START:]
    out = {"n": int(x.dropna().shape[0]), "mae_forecast_vs_actual_g_per_kwh": float((x.a - x.f).abs().mean())}
    for name, lag in (("persistence_30min", 1), ("persistence_3h", 6), ("persistence_12h", 24), ("persistence_24h", 48)):
        out[f"mae_{name}_g_per_kwh"] = float((a - a.shift(lag))[ad.TEST_START:].abs().mean())
    out["note"] = ("The API does not document the issue time of the archived forecast. An error far below 24 h persistence "
                   "is consistent with a forecast issued closer to the valid time than one day; the ablation is conditional "
                   "on the assumption that the value for time v was available at v minus 24 h.")
    return out


def ckpt_projection_diagnostic(d_nom: float, lead_s: float = 20.0) -> dict:
    """Timing error (s) of the projected next checkpoint against the logged one, for run2, at origins `lead_s` earlier."""
    from heatwave.data.schema import load_series

    df, meta = load_series("laptop_run2_1s")
    ev = load_series("laptop_run2_events")[0]
    sched = cv.laptop_schedule_from_events(meta["source"]["schedule"], ev, d_nom)
    errs = []
    for s in sched.ckpt_start:
        origin = int(s - lead_s * 1e9)
        if origin <= sched.seg_start[0]:
            continue
        cand = cv.laptop_ckpt_starts_future(sched, origin, int(2 * lead_s * 1e9))
        if cand:
            errs.append(min(((c - s) / 1e9 for c in cand), key=abs))
    e = np.abs(np.array(errs))
    return {"n_logged_checkpoints": int(len(sched.ckpt_start)), "n_with_projection": int(len(e)),
            "lead_s": lead_s, "nominal_write_duration_s": d_nom,
            "abs_timing_error_s": {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "max": float(e.max())} if len(e) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="dataset names to run (default all)")
    ap.add_argument("--sizes", nargs="+", default=["209M", "28M"])
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stride-mult", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--cache-dir", default=str(ac.ROOT / "results" / "ablation" / "cache"))
    ap.add_argument("--out", default=str(ac.ROOT / "results" / "ablation" / "ablation_real.json"))
    ap.add_argument("--run-label", default="")
    ap.add_argument("--no-const-guard", action="store_true", help="feed constant non-binary covariate rows to the model unchanged")
    a = ap.parse_args()

    forecasters = {s: ac.make_timestral(s, a.weights_dir, a.device, a.threads) for s in a.sizes}
    d_nom = ad.laptop_d_nominal("run1")
    todo = []
    for name, (ds, cfgs, labels) in ad.build_real(a.stride_mult).items():
        todo.append((ds, cfgs, labels))
    for run in ("run2", "run1"):
        for q, fr in LAPTOP_SETS:
            todo.append(ad.build_laptop(run, fr, q, d_nom, a.stride_mult))
    if a.only:
        todo = [t for t in todo if t[0].name in a.only]
    out_path = Path(a.out)
    results, t_all = {}, time.time()
    design = {
        "split": "grid and carbon series: origins from 2026-05-01 (fixed test split); laptop: run2 = test, run1 = calibration run "
                 "(zero-shot model: nothing is fitted on either)",
        "leakage_rules": {
            "DA": "CAISO day-ahead LMP for local day D is revealed from 13:00 Pacific on D-1 (assumed publication time); NaN before",
            "WX": "Open-Meteo previous_day1 forecast: the value for valid time v is revealed from v minus 24 h; NaN beyond",
            "DEM": "CAISO Outlook day-ahead demand forecast; same publication rule as DA (assumed)",
            "F": "GB archived intensity forecast; vintage undocumented; revealed from v minus 24 h (assumed)",
            "S": "schedule segments plus the observed start of the segment running at the origin",
            "K": "next checkpoint = end of the last observed checkpoint + the segment's period; nominal write duration from run1",
            "L_realised": "logged checkpoint intervals of the whole run (realised; NOT known at the origin; comparison only)",
        },
        "bootstrap": f"moving-block bootstrap over origins (block = ceil(horizon / origin stride)), paired across configs, {a.reps} replicates, 95% percentile intervals",
        "const_guard": not a.no_const_guard,
        "const_guard_rule": "covariate rows (window, channel) that are exactly constant and not 0/1 are replaced by zeros before the model; counts in constant_rows_zeroed",
        "sizes": a.sizes, "weights_dir": a.weights_dir, "stride_mult": a.stride_mult,
        "laptop_ckpt_nominal_duration_s": d_nom,
    }
    meta = {"hardware": ac.hardware_info(a.device), "run_label": a.run_label}
    diagnostics = {"gb_forecast_vintage": gb_vintage_diagnostic(), "laptop_ckpt_projection_run2": ckpt_projection_diagnostic(d_nom)}
    inputs = sorted(Path(ac.ROOT / "data" / "processed").glob("laptop_run*_1s.parquet"))
    for ds, cfgs, labels in todo:
        t0 = time.time()
        methods = [MethodSpec(f.name, f, cfgs) for f in forecasters.values()] + [
            MethodSpec(f"seasonal-naive-{ds.scale.season}", SeasonalNaiveForecaster(ds.scale.season), [("none", ())], False)]
        ac.log(f"{ds.name}: {ds.n_origins} origins, configs {[c for c, _ in cfgs]}")
        run = evaluate(ds, methods, batch_size=a.batch_size, log=ac.log, const_guard=not a.no_const_guard)
        block = result_block(ds, run, methods, reps=a.reps, seed=0, cluster=None)
        np.savez_compressed(Path(a.cache_dir) / f"stats_{ds.name}.npz",
                            **{f"{m}|{c}": v for m, d in run["stats"].items() for c, v in d.items()})
        results[ds.name] = {
            "data_kind": ds.data_kind, "what": ds.what, "covariate_groups": labels,
            "scale": {"context": ds.scale.context, "horizon": ds.scale.horizon, "step": ds.scale.step,
                      "buckets": ds.scale.labels(), "season": ds.scale.season},
            "subsampling": ds.subsampling(), "methods": block, "wall_seconds": time.time() - t0,
        }
        write_result(out_path, {"design": design, "meta": meta, "diagnostics": diagnostics, "datasets": results, "complete": False},
                     script=__file__, data_kind="mixed", inputs=inputs, notes=f"partial; run_label={a.run_label}")
    write_result(out_path, {"design": design, "meta": {**meta, "wall_seconds_total": time.time() - t_all},
                            "diagnostics": diagnostics, "datasets": results, "complete": True},
                 script=__file__, data_kind="mixed", inputs=inputs,
                 notes=f"public datasets and MEASURED laptop runs (per-dataset data_kind inside). run_label={a.run_label}. Hardware: {meta['hardware']}")
    ac.log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
