"""Run the SIMULATED gpu_stack heatwave scenario (HEATWAVE_SIM_SCENARIO, default sim-scenario-v3) for the given seeds and
cache the arrays that the forecasting studies read (results/cache/forecast/sim_<version>_seed<S>.npz, gitignored).

This is data preparation, not a study: one seven-day run takes about three minutes per seed on one core, so seeds are
run in parallel processes. Default seeds: TEST 100-109 (multi-scale and hierarchy evaluation) and validation 50-53
(estimating the MinT covariance). Existing cache files are kept unless --force.

Usage: python experiments/forecast/sim_cache.py [--seeds 100,...,109,50,...,53] [--workers 2] [--horizon-s N]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from heatwave.forecast import datasets as D  # noqa: E402


def one(args):
    seed, horizon = args
    t = time.time()
    p = D.build_sim_cache(seed, horizon_s=horizon)
    return seed, str(p), time.time() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in list(D.SIM_TEST_SEEDS) + list(D.SIM_VAL_SEEDS)))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--horizon-s", type=int, default=0, help="shorter horizon for development (0 = scenario default, 7 days)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]
    todo = [(s, a.horizon_s or None) for s in seeds if a.force or not D.sim_cache_path(s).exists()]
    print(f"{len(todo)} of {len(seeds)} seeds to run", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for seed, path, sec in ex.map(one, todo):
            print(f"seed {seed}: {path} ({sec:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
