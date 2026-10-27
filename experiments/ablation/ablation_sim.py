"""Covariate ablation on the SIMULATED e003 cluster with a PUBLIC environment (data kind: mixed).

Forecaster: Timestral zero-shot (or a fine-tuned checkpoint via --weights-dir), plus a seasonal-naive reference.
Series (test seeds 100-109): site power (hourly and 1-min), rack temperature (1-min), rack power (1 s).
Covariate groups: J job schedule (planned GPUs running, planned checkpoint GPUs), P training-step phase,
W weather forecast (Open-Meteo archived day-ahead vintage), C day-ahead price and seasonal-naive carbon forecast.
Configurations: none; each group alone; forward addition in the pre-declared order J, P, W, C; all four; leave one out.
Scores: WQL, MASE, 80% and 98% coverage per horizon bucket; delta WQL against `none` with paired block-bootstrap CIs.

The simulator cache comes from sim_cache.py (scenario sim-scenario-v3: the job schedule follows the run seed; the
plant runs on public weather, price and carbon via heatwave.sim_env.public_environment, one week per seed).

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
    ap.add_argument("--usd-per-eur", type=float, default=None, help="default: read from results/cost-v1.json")
    ap.add_argument("--run-label", default="")
    ap.add_argument("--resume", action="store_true",
                    help="skip datasets already present in --out (partial file) and resume the unfinished one from its checkpoint")
    ap.add_argument("--checkpoint-dir", default=None, help="per-dataset checkpoints (default: <cache-dir>/ckpt)")
    ap.add_argument("--checkpoint-every", type=int, default=25, help="batches between checkpoints")
    ap.add_argument("--no-const-guard", action="store_true", help="feed constant non-binary covariate rows to the model unchanged")
    ap.add_argument("--guard", choices=["exact", "context"], default="exact",
                    help="exact: zero rows that are exactly constant; context: zero rows whose context part is constant")
    a = ap.parse_args()

    scenario = json.loads(Path(cv.SCENARIO_PATH).read_text())
    if a.usd_per_eur is None:
        usd_per_eur, fx_src = cv.usd_per_eur_from_cost_result()
    else:
        usd_per_eur, fx_src = a.usd_per_eur, "command line"
    configs, roles = ac.sim_configs()
    forecasters = {s: ac.make_timestral(s, a.weights_dir, a.device, a.threads) for s in a.sizes}
    out_path = Path(a.out)
    results: dict = {}
    if a.resume and out_path.exists():
        prev = json.loads(out_path.read_text())["result"]
        if not prev.get("complete"):
            results = prev["datasets"]
        ac.log(f"resume: datasets already complete in {out_path}: {list(results)}")
    ckpt_dir = Path(a.checkpoint_dir or Path(a.cache_dir) / "ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    design = {
        "forward_order": list(ac.FORWARD_ORDER),
        "groups": {g: {"name": cv.SIM_GROUP_NAMES[g], "channels": list(cv.SIM_GROUPS[g])} for g in cv.SIM_GROUPS},
        "configs": {n: {"groups": list(g), "roles": roles[n]} for n, g in configs},
        "scenario": {"id": scenario["scenario_id"], "path": str(cv.SCENARIO_PATH),
                     "note": "job schedule follows the run seed; the scenario synthetic environment is replaced by the public environment; plant SIMULATED"},
        "environment": {
            "kind": "public weather, day-ahead price and carbon via heatwave.sim_env.public_environment (mixed: simulated plant, public exogenous series)",
            "week_start_rule": "seed s runs the week starting at scenario start_utc + 7 x (s - 100) days (all weeks in the test period, 2026-05-01 onward)",
            "usd_per_eur": usd_per_eur, "usd_per_eur_source": fx_src,
            "sites": {"site-a": "Bay Area weather, CAISO NP15 day-ahead LMP, CAISO carbon", "site-b": "Frankfurt weather, DE-LU day-ahead price x usd_per_eur, DE carbon"},
        },
        "leakage_rules": {
            "J": "plan built from jobs with submit_s <= origin only; horizon <= known_future_horizon_s",
            "P": "same plan as J (phase of the planned steps)",
            "W": "Open-Meteo previous_day1 (day-ahead) forecast, available from valid time minus 24 h; never the actual (ERA5-based) outdoor temperature the plant used",
            "C": "day-ahead price, available from 13:00 local market time on the day before delivery (assumed publication time; unpublished steps are NaN); "
                 "carbon: seasonal-naive forecast (actual of 24 h earlier, available 23 h before its hour); the simulator's actual-carbon export is not used",
        },
        "bootstrap": "moving-block bootstrap over origins (block = ceil(horizon / origin stride)), same resampled origins for "
                     "every stream and config (paired); cluster bootstrap over seeds in delta_wql_cluster_ci95; "
                     f"{a.reps} replicates, 95% percentile intervals",
        "const_guard": not a.no_const_guard,
        "const_guard_rule": ("covariate rows (window, channel) that are exactly constant and not 0/1 are replaced by zeros before the model; counts in constant_rows_zeroed"
                             if a.guard == "exact" else
                             "covariate rows (window, channel) whose context part is constant and that are not 0/1 indicators are replaced by zeros before the model "
                             "(no in-context evidence of a relation to the series); counts in constant_rows_zeroed"),
        "guard": a.guard,
        "sizes": a.sizes, "weights_dir": a.weights_dir, "seeds": a.seeds, "stride_mult": a.stride_mult,
        "split": "simulator test seeds 100-109 (fixed split; no fitting on any series)",
    }
    meta = {"hardware": ac.hardware_info(a.device), "run_label": a.run_label}
    for _ in [0]:
        for kind in a.kinds:
            if SIM_KINDS[kind]["scale"].name in results:
                continue
            t0 = time.time()
            ds = SimDataset(kind, a.seeds, Path(a.cache_dir), scenario, usd_per_eur, stride_mult=a.stride_mult)
            methods = [
                MethodSpec(f.name, f, configs) for f in forecasters.values()
            ] + [MethodSpec(f"seasonal-naive-{ds.scale.season}", SeasonalNaiveForecaster(ds.scale.season), [("none", ())], False)]
            ac.log(f"{ds.name}: {ds.n_streams} streams x {ds.n_origins} origins, methods {[m.name for m in methods]}")
            run = evaluate(ds, methods, batch_size=a.batch_size, log=ac.log, const_guard=not a.no_const_guard, guard=a.guard,
                           checkpoint=str(ckpt_dir / f"{ds.name}.npy"), checkpoint_every=a.checkpoint_every)
            block = result_block(ds, run, methods, reps=a.reps, seed=0, cluster=ds.cluster)
            results[ds.name] = {
                "data_kind": "mixed (simulated plant, public environment)", "quantity": kind, "what": ds.what,
                "scale": {"context": ds.scale.context, "horizon": ds.scale.horizon, "step": ds.scale.step,
                          "buckets": ds.scale.labels(), "season": ds.scale.season},
                "subsampling": ds.subsampling(), "methods": block, "wall_seconds": time.time() - t0,
            }
            np.savez_compressed(
                Path(a.cache_dir) / f"stats_{ds.scale.name}.npz",
                **{f"{m}|{c}": v for m, d in run["stats"].items() for c, v in d.items()},
            )
            content = {"design": design, "meta": meta, "datasets": results, "complete": False}
            write_result(out_path, content, script=__file__, data_kind="mixed",
                         inputs=sorted(Path(a.cache_dir).glob("sim_v3_*.npz")),
                         notes=f"SIMULATED plant (e003 sim-scenario-v3) with PUBLIC weather, price and carbon (mixed). run_label={a.run_label}; partial={True}")
    content = {"design": design, "meta": {**meta, "wall_seconds_total": time.time() - t_all},
               "datasets": results, "complete": True}
    write_result(out_path, content, script=__file__, data_kind="mixed",
                 inputs=sorted(Path(a.cache_dir).glob("sim_v3_*.npz")),
                 notes=f"SIMULATED plant (e003 sim-scenario-v3) with PUBLIC weather, price and carbon (mixed). run_label={a.run_label}. Hardware: {meta['hardware']}")
    ac.log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
