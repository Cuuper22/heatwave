"""Where does a covariate's extra loss come from? Joins per-window losses with the shape of the covariate rows.

For one simulator dataset and seed, every window is classed by each row of a covariate group: constant over the
context or varying in it, and changing inside the horizon or not. The script then reports, per class, the loss with
the group against the loss without it and the share of the group's extra loss. Per-window losses come from the
evaluation checkpoint that ablation_sim.py writes (results/ablation/cache/ckpt/<dataset>.npy).

  python experiments/ablation/diagnose_horizon_steps.py --checkpoint <ckpt.npy> --kind rack_power_1s --seed 100 \\
      --groups C J W --out results/ablation/diagnosis_horizon_steps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import abl_common as ac
from abl_data import SimDataset
from abl_engine import guard_constant_rows
from heatwave.forecast import covariates as cv
from heatwave.provenance import write_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kind", default="rack_power_1s")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--seeds-in-run", type=int, nargs="+", default=list(range(100, 110)))
    ap.add_argument("--groups", nargs="+", default=["C", "J", "W"])
    ap.add_argument("--sizes", nargs="+", default=["timestral-209M", "timestral-28M"])
    ap.add_argument("--cache-dir", default=str(ac.ROOT / "results" / "ablation" / "cache"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = np.load(a.checkpoint, allow_pickle=True)[0]
    st = ck["stats"]
    scenario = json.loads(Path(cv.SCENARIO_PATH).read_text())
    fx, _ = cv.usd_per_eur_from_cost_result()
    ds = SimDataset(a.kind, [a.seed], Path(a.cache_dir), scenario, fx)
    offset = a.seeds_in_run.index(a.seed) * len(ds.units)  # streams are seed-major in the full run
    T = ds.scale.context
    out = {}
    for g in a.groups:
        classes: dict = {}
        for b in ds.iter_batches(64):
            c, _ = guard_constant_rows(b.cov[g])
            for i in range(len(b.stream)):
                s, o = int(b.stream[i]) + offset, int(b.origin[i])
                losses = {m: (float(st[m]["none"][s, o, -1, 0]), float(st[m][g][s, o, -1, 0])) for m in a.sizes}
                if not all(np.isfinite(v).all() for v in losses.values()):
                    continue
                tags = []
                for k in range(c.shape[1]):
                    r = np.nan_to_num(c[i, k], nan=-9e9)
                    tags.append(("context varies" if np.unique(r[:T]).size > 1 else "context constant")
                                + (", changes in horizon" if np.unique(r[T - 1:]).size > 1 else ""))
                classes.setdefault(" / ".join(tags), []).append(losses)
        res = {}
        for m in a.sizes:
            tot = sum(w[m][1] - w[m][0] for v in classes.values() for w in v)
            res[m] = {k: {"n_windows": len(v), "loss_with_over_without": sum(w[m][1] for w in v) / max(sum(w[m][0] for w in v), 1e-12),
                          "share_of_extra_loss": (sum(w[m][1] - w[m][0] for w in v) / tot) if tot else None}
                      for k, v in sorted(classes.items(), key=lambda kv: -len(kv[1]))}
            res[m]["_extra_loss_total"] = tot
        out[g] = {"channels": list(cv.SIM_GROUPS[g]), "by_class": res}
        ac.log(f"{g}: {json.dumps(res['timestral-209M'])[:400]}")
    write_result(Path(a.out), {"dataset": ds.name, "seed": a.seed, "loss": "per-window pinball loss sum, all horizons",
                               "groups": out}, script=__file__, data_kind="mixed",
                 inputs=[Path(a.checkpoint)], notes="SIMULATED plant with PUBLIC environment (mixed); one seed; diagnosis only")


if __name__ == "__main__":
    main()
