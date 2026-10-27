"""Run the e003 Heatwave simulator (SIMULATED plant, PUBLIC environment) for one seed and cache the series the ablation needs.

Scenario: gpu_stack branch `heatwave`, experiments/e003-heatwave/sim-scenario-v3.json (job schedule follows the run
seed). The environment is public: Open-Meteo weather (actual for the plant, archived day-ahead forecast for the
covariate), CAISO and DE-LU day-ahead prices (DE converted at the ECB mean rate in results/cost-v1.json), CAISO and DE
carbon intensity (`heatwave.sim_env.public_environment`). Seed s runs the week starting at the scenario start_utc
plus 7 x (s - 100) days.

Usage: python experiments/ablation/sim_cache.py --seeds 100 101 ... [--out results/ablation/cache]
Output: <out>/sim_v3_<seed>.npz (gitignored cache).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from heatwave.forecast.covariates import (  # noqa: E402
    SCENARIO_PATH, public_env_for_seed, seed_start_utc, usd_per_eur_from_cost_result,
)

JSON_ENC = json.JSONEncoder(sort_keys=True)


def run_one(seed: int, out_dir: Path, usd_per_eur: float) -> Path:
    from gpu_stack.research import heatwave_sim as hs

    sc = hs.load_scenario(SCENARIO_PATH)
    sc["outputs"] = {"gpu_windows": []}
    pe = public_env_for_seed(sc, seed, usd_per_eur)
    t0 = time.time()
    res = hs.run_scenario(sc, seed=seed, environment=pe.environment)
    a = res.to_arrays()
    site_1min = a["site_power_w"].reshape(-1, 60, a["site_power_w"].shape[1]).mean(axis=1)
    meta = a["meta"]
    out = out_dir / f"sim_v3_{seed}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "scenario_id": sc["scenario_id"], "start_utc": str(seed_start_utc(sc, seed)), "usd_per_eur": usd_per_eur,
        "filled_share": pe.filled_share, "sources": pe.sources,
    }
    np.savez_compressed(
        out,
        site_mean_power_w_1h=a["site_mean_power_w_1h"],
        site_power_w_1min=site_1min,
        rack_power_w_1s=a["rack_power_w"].astype(np.float32),
        rack_temp_c_1min=a["rack_temp_c_1min"],
        rack_ids=np.array(meta["rack_ids"]),
        site_ids=np.array(meta["site_ids"]),
        seed=np.array(seed),
        env_info=np.array(JSON_ENC.encode(info)),
        wall_s=np.array(time.time() - t0),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "ablation" / "cache"))
    ap.add_argument("--usd-per-eur", type=float, default=None, help="default: read from results/cost-v1.json")
    a = ap.parse_args()
    fx = a.usd_per_eur if a.usd_per_eur is not None else usd_per_eur_from_cost_result()[0]
    for s in a.seeds:
        t = time.time()
        p = run_one(s, Path(a.out), fx)
        print(f"seed {s}: {p} ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
