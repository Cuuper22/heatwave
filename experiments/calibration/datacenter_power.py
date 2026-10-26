"""Calibrate the e003 simulator's GPU power levels and GPU thermal offset on PUBLIC H100 data.

Source: Elsayed, Al-Obaidi, Farag (2026), Scientific Data 13:1268,
doi:10.1038/s41597-026-07496-6, data doi:10.6084/m9.figshare.31654879 (CC BY 4.0):
8x H100 SXM node, LLM training with DeepSpeed, per-GPU power/utilization/
temperature at 20 ms (and 1 s bins, see data/SOURCES.md).

Per GPU and session (H100, llm_training), from the 20 ms series:
  idle_w      1st percentile of power (the floor the GPU returns to)
  compute_w   median power over 1 s bins with utilization >= 95 %
  low_w       median power of 20 ms samples inside the active window that
              sit below the midpoint between idle_w and compute_w (the
              communication / waiting part of a step)
  low_frac    share of active-window 20 ms samples in that low state
  limit_w     power / (power as % of limit), the board limit
Per GPU and session, from 1 s bins: least-squares slope of temperature on
power (K/W) during the active window, the steady GPU-to-air offset the
simulator calls gpu_to_air_k_per_w, plus a 1-node RC fit (heat capacity,
resistance, time constant) with gpu_stack.research.heatwave_thermal.fit.

Also reads the MEASURED laptop checkpoint/compute ratio from the energy
counter (results of scripts/process_laptop_run.py), labeled as laptop data.

Writes results/calibration/datacenter_power.json (data_kind="mixed": public
H100 data plus measured laptop data).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from heatwave.data import public as P  # noqa: E402
from heatwave.data.schema import PROCESSED, load_series  # noqa: E402
from heatwave.provenance import write_result  # noqa: E402

from gpu_stack.research import heatwave_thermal as ht  # noqa: E402


def active_window(util_1s, thresh=50.0):
    on = np.flatnonzero(util_1s >= thresh)
    return (on[0], on[-1]) if len(on) else (None, None)


def per_gpu(sess: str, k: int) -> dict | None:
    d20 = P.load_node_power(sess, freq="20ms")
    d1 = P.load_node_power(sess, freq="1s")
    p20 = d20[f"g{k}_power_w"]
    p1, u1, t1 = d1[f"g{k}_power_w"], d1[f"g{k}_util_pct"], d1[f"g{k}_temp_c"]
    a, b = active_window(u1.to_numpy())
    if a is None or b - a < 120:
        return None
    t_a, t_b = d1.index[a], d1.index[b]
    idle = float(np.nanpercentile(p20, 1))
    comp = float(p1[u1 >= 95].median())
    act = p20[(p20.index >= t_a) & (p20.index <= t_b)].dropna()
    mid = 0.5 * (idle + comp)
    low = act[act < mid]
    lim = float((p1 / (d1[f"g{k}_power_pct_limit"] / 100.0)).median())
    seg = slice(a, b + 1)
    pw, tc = p1.iloc[seg].to_numpy(), t1.iloc[seg].to_numpy()
    ok = np.isfinite(pw) & np.isfinite(tc)
    slope = float(np.polyfit(pw[ok], tc[ok], 1)[0]) if ok.sum() > 30 else float("nan")
    # RC fit over the whole session (1 s), ambient folded into the setpoint.
    P_all = p1.interpolate(limit_direction="both").to_numpy()
    T_all = t1.interpolate(limit_direction="both").to_numpy()
    try:
        rc, rep = ht.fit(P_all, np.zeros_like(P_all), T_all, 1.0,
                         initial={"heat_capacity_j_per_k": 500.0, "resistance_k_per_w": 0.05,
                                  "supply_setpoint_c": float(T_all.min())})
        rc_d = {"heat_capacity_j_per_k": rc.heat_capacity_j_per_k,
                "resistance_k_per_w": rc.resistance_k_per_w,
                "time_constant_s": rc.time_constant_s, "rmse_c": rep["rmse_c"], "r2": rep["r2"]}
    except Exception as e:  # keep the session, report the failure
        rc_d = {"error": str(e)}
    return {"idle_w": idle, "compute_w": comp, "low_w": float(low.median()) if len(low) else float("nan"),
            "low_frac": float(len(low) / len(act)) if len(act) else float("nan"),
            "limit_w": lim, "temp_per_power_k_per_w": slope, "rc": rc_d}


def laptop_checkpoint_ratio(run: str = "run1") -> dict:
    df, _ = load_series(f"laptop_{run}_10hz")
    ck = df["ckpt_scheduled"] > 0
    tr = (df["training_on"] > 0) & ~ck
    c, t = float(df.loc[ck, "power_energy_w"].median()), float(df.loc[tr, "power_energy_w"].median())
    return {"run": run, "checkpoint_power_w": c, "training_power_w": t, "ratio": c / t,
            "n_checkpoint_samples": int(ck.sum()),
            "signal": "power_energy_w (NVML energy counter, about 94 ms updates)",
            "device": "RTX 3060 Laptop GPU (measured; a laptop checkpoint of a 27M-parameter model to local disk)"}


def summarize(rows: list[dict], key: str) -> dict:
    v = np.array([r[key] for r in rows if np.isfinite(r[key])])
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "n": int(len(v))}


def main() -> None:
    s = P.list_node_power_sessions()
    h = s[(s.hardware == "H100") & (s.task == "llm_training")]
    rows, per_session = [], {}
    for sess in h.session:
        rs = []
        for k in range(8):
            r = per_gpu(sess, k)
            if r is not None:
                r["session"], r["gpu"] = sess, k
                rs.append(r)
        per_session[sess] = rs
        rows += rs
    limit = summarize(rows, "limit_w")["median"]
    for r in rows:
        r["idle_ratio"] = r["idle_w"] / r["compute_w"]
        r["compute_fraction_of_limit"] = r["compute_w"] / r["limit_w"]
        r["low_ratio"] = r["low_w"] / r["compute_w"]
    rc_ok = [r["rc"] for r in rows if "r2" in r["rc"] and r["rc"]["r2"] > 0.5]
    res = {
        "source": "Elsayed et al. 2026 (doi:10.1038/s41597-026-07496-6), H100 SXM 8-GPU node, llm_training sessions",
        "n_sessions": int(len(h)), "n_gpu_sessions": len(rows),
        "limit_w_median": limit,
        "idle_w": summarize(rows, "idle_w"),
        "compute_w": summarize(rows, "compute_w"),
        "idle_ratio": summarize(rows, "idle_ratio"),
        "compute_fraction_of_limit": summarize(rows, "compute_fraction_of_limit"),
        "low_phase_ratio": summarize(rows, "low_ratio"),
        "low_phase_fraction_of_active_time": summarize(rows, "low_frac"),
        "temp_per_power_k_per_w": summarize(rows, "temp_per_power_k_per_w"),
        "gpu_rc_fit": {
            "n_fits_r2_gt_0.5": len(rc_ok), "n_attempted": len(rows),
            "time_constant_s": summarize(rc_ok, "time_constant_s") if rc_ok else None,
            "resistance_k_per_w": summarize(rc_ok, "resistance_k_per_w") if rc_ok else None,
            "heat_capacity_j_per_k": summarize(rc_ok, "heat_capacity_j_per_k") if rc_ok else None,
            "r2": summarize(rc_ok, "r2") if rc_ok else None,
        },
        "laptop_checkpoint": laptop_checkpoint_ratio(),
        "per_gpu": rows,
    }
    inputs = sorted((PROCESSED / "public_gpu_node_20ms").glob("LLM_H100*.parquet"))
    inputs += [PROCESSED / "laptop_run1_10hz.parquet"]
    write_result(ROOT / "results/calibration/datacenter_power.json", res, script=__file__,
                 data_kind="mixed", inputs=inputs,
                 notes="public H100 data (CC BY 4.0) plus measured laptop checkpoint ratio")
    for k in ("limit_w_median", "idle_w", "compute_w", "idle_ratio", "compute_fraction_of_limit",
              "low_phase_ratio", "low_phase_fraction_of_active_time", "temp_per_power_k_per_w", "gpu_rc_fit"):
        print(k, res[k])
    print("laptop_checkpoint", {k: res["laptop_checkpoint"][k] for k in ("ratio", "checkpoint_power_w", "training_power_w")})


if __name__ == "__main__":
    main()
