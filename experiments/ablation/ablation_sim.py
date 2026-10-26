"""Covariate ablation on the SIMULATED e003 cluster (all four known-future covariates exist).

Forecaster: Timestral zero-shot (or a fine-tuned checkpoint via --weights-dir), plus a seasonal-naive reference.
Series (test seeds 100-109): site power (hourly and 1-min), rack temperature (1-min), rack power (1 s).
Covariate groups: J job schedule (planned GPUs running, planned checkpoint GPUs), P training-step phase,
W weather (outdoor temperature) forecast, C price and carbon forecast.
Configurations: none; each group alone; forward addition in the pre-declared order J, P, W, C; all four; leave one out.
Scores: WQL, MASE, 80% and 98% coverage per horizon bucket; delta WQL against `none` with paired block-bootstrap CIs.

The simulator cache comes from sim_cache.py. Variants: `literal` (scenario as committed: all seeds share one job
schedule and weather path) and `worlds` (job and environment seeds follow the run seed).

Usage (development, 2 cores):
  python experiments/ablation/ablation_sim.py --sizes 5M --seeds 100 101 --stride-mult 6 --reps 200 --out results/ablation/dev/sim.json
Usage (Colab A100, full):
  python experiments/ablation/ablation_sim.py --sizes 209M 28M --device cuda --out results/ablation/ablation_sim.json
Fine-tuned checkpoint: add --weights-dir <dir> (name becomes timestral-<size>-finetuned).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import abl_common as ac
from abl_data import SIM_KINDS, SimDataset
from abl_engine import MethodSpec, evaluate, result_block
from heatwave.forecast import SeasonalNaiveForecaster
from heatwave.forecast import covariates as cv
from heatwave.provenance import write_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["literal", "worlds"], choices=["literal", "worlds"])
    ap.add_argument("--kinds", nargs="+", default=list(SIM_KINDS), choices=list(SIM_KINDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    ap.add_argument("--sizes", nargs="+", default=["209M", "28M"])
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stride-mult", type=int, default=1, help="multiply the origin stride (subsampling)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--cache-dir", default=str(ac.ROOT / "results" / "ablation" / "cache"))
    ap.add_argument("--out", default=str(ac.ROOT / "results" / "ablation" / "ablation_sim.json"))
    ap.add_argument("--run-label", default="")
    ap.add_argument("--no-const-guard", action="store_true", help="feed constant non-binary covariate rows to the model unchanged")
    a = ap.parse_args()

    scenario = json.loads(Path(cv.SCENARIO_PATH).read_text())
    configs, roles = ac.sim_configs()
    forecasters = {s: ac.make_timestral(s, a.weights_dir, a.device, a.threads) for s in a.sizes}
    out_path = Path(a.out)
    results: dict = {}
    t_all = time.time()
    design = {
        "forward_order": list(ac.FORWARD_ORDER),
        "groups": {g: {"name": cv.SIM_GROUP_NAMES[g], "channels": list(cv.SIM_GROUPS[g])} for g in cv.SIM_GROUPS},
        "configs": {n: {"groups": list(g), "roles": roles[n]} for n, g in configs},
        "variants": {
            "literal": "scenario as committed; the job-generator seed (7) and environment seed (11) are fixed in the file, "
                       "so seeds 100-109 share one job schedule, weather, price and carbon path and differ in noise only",
            "worlds": "job-generator seed and environment seed set to the run seed: each seed is a different synthetic "
                      "job schedule, weather, price and carbon path; fault and grid events keep their scheduled times",
        },
        "leakage_rules": {
            "J": "plan built from jobs with submit_s <= origin only; horizon <= known_future_horizon_s",
            "P": "same plan as J (phase of the planned steps)",
            "W": "scenario outdoor *forecast* series (simulated: truth + AR(1) error, sigma 1.2 degC), never the actual outdoor temperature",
            "C": "scenario price and carbon series passed through without error (day-ahead style); SIMULATED",
        },
        "bootstrap": "moving-block bootstrap over origins (block = ceil(horizon / origin stride)), same resampled origins for "
                     "every stream and config (paired); cluster bootstrap over seeds in delta_wql_cluster_ci95; "
                     f"{a.reps} replicates, 95% percentile intervals",
        "const_guard": not a.no_const_guard,
        "const_guard_rule": "covariate rows (window, channel) that are exactly constant and not 0/1 are replaced by zeros before the model; counts in constant_rows_zeroed",
        "sizes": a.sizes, "weights_dir": a.weights_dir, "seeds": a.seeds, "stride_mult": a.stride_mult,
        "split": "simulator test seeds 100-109 (fixed split; no fitting on any series)",
    }
    meta = {"hardware": ac.hardware_info(a.device), "run_label": a.run_label}
    for variant in a.variants:
        for kind in a.kinds:
            t0 = time.time()
            ds = SimDataset(kind, variant, a.seeds, Path(a.cache_dir), scenario, stride_mult=a.stride_mult)
            methods = [
                MethodSpec(f.name, f, configs) for f in forecasters.values()
            ] + [MethodSpec(f"seasonal-naive-{ds.scale.season}", SeasonalNaiveForecaster(ds.scale.season), [("none", ())], False)]
            ac.log(f"{ds.name}: {ds.n_streams} streams x {ds.n_origins} origins, methods {[m.name for m in methods]}")
            run = evaluate(ds, methods, batch_size=a.batch_size, log=ac.log, const_guard=not a.no_const_guard)
            block = result_block(ds, run, methods, reps=a.reps, seed=0, cluster=ds.cluster)
            results[ds.name] = {
                "data_kind": "simulated", "variant": variant, "quantity": kind, "what": ds.what,
                "scale": {"context": ds.scale.context, "horizon": ds.scale.horizon, "step": ds.scale.step,
                          "buckets": ds.scale.labels(), "season": ds.scale.season},
                "subsampling": ds.subsampling(), "methods": block, "wall_seconds": time.time() - t0,
            }
            np.savez_compressed(
                Path(a.cache_dir) / f"stats_{ds.scale.name}_{variant}.npz",
                **{f"{m}|{c}": v for m, d in run["stats"].items() for c, v in d.items()},
            )
            content = {"design": design, "meta": meta, "datasets": results, "complete": False}
            write_result(out_path, content, script=__file__, data_kind="simulated",
                         inputs=sorted(Path(a.cache_dir).glob("sim_*_*.npz")),
                         notes=f"SIMULATED data (e003 sim-scenario-v2). run_label={a.run_label}; partial={True}")
    content = {"design": design, "meta": {**meta, "wall_seconds_total": time.time() - t_all},
               "datasets": results, "complete": True}
    write_result(out_path, content, script=__file__, data_kind="simulated",
                 inputs=sorted(Path(a.cache_dir).glob("sim_*_*.npz")),
                 notes=f"SIMULATED data (e003 sim-scenario-v2). run_label={a.run_label}. Hardware: {meta['hardware']}")
    ac.log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
