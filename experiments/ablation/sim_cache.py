"""Run the e003 Heatwave simulator (SIMULATED data) for one seed and cache the series the ablation needs.

Scenario: gpu_stack branch `heatwave`, experiments/e003-heatwave/sim-scenario-v2.json.

Variants (see heatwave/forecast/covariates.py, `world_scenario`):
  literal : the committed scenario unchanged; the run seed changes only the noise streams, because the scenario file fixes
            the job-generator seed (7) and the environment seed (11). All seeds share one job schedule and one weather path.
  worlds  : job-generator seed and environment seed set to the run seed, so each seed is a different synthetic job
            schedule, weather, price and carbon path (fault and grid events keep their scheduled times).

Usage: python experiments/ablation/sim_cache.py --variant literal|worlds --seeds 100 101 ... [--out results/ablation/cache]
Output: <out>/sim_<variant>_<seed>.npz (gitignored cache).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from heatwave.forecast.covariates import SCENARIO_PATH, world_scenario  # noqa: E402


def run_one(variant: str, seed: int, out_dir: Path) -> Path:
    from gpu_stack.research import heatwave_sim as hs

    sc = hs.load_scenario(SCENARIO_PATH)
    sc = world_scenario(sc, seed, variant)
    sc["outputs"] = {"gpu_windows": []}
    t0 = time.time()
    res = hs.run_scenario(sc, seed=seed)
    a = res.to_arrays()
    site_1min = a["site_power_w"].reshape(-1, 60, a["site_power_w"].shape[1]).mean(axis=1)
    meta = a["meta"]
    out = out_dir / f"sim_{variant}_{seed}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        site_mean_power_w_1h=a["site_mean_power_w_1h"],
        site_power_w_1min=site_1min,
        rack_power_w_1s=a["rack_power_w"].astype(np.float32),
        rack_temp_c_1min=a["rack_temp_c_1min"],
        rack_ids=np.array(meta["rack_ids"]),
        site_ids=np.array(meta["site_ids"]),
        seed=np.array(seed),
        variant=np.array(variant),
        wall_s=np.array(time.time() - t0),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["literal", "worlds"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "ablation" / "cache"))
    a = ap.parse_args()
    for s in a.seeds:
        t = time.time()
        p = run_one(a.variant, s, Path(a.out))
        print(f"{a.variant} seed {s}: {p} ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
