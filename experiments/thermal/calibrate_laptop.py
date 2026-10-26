"""Calibrate RC thermal models on MEASURED laptop GPU runs.

Fits on one run and validates on another (out of sample):
  * 1-node RC: gpu_stack.research.heatwave_thermal.fit (the rack model used in
    the e003 simulator), with the supply setpoint free (no ambient sensor on
    the laptop, so ambient is folded into the setpoint).
  * 2-node RC (die + heatsink), fitted here with scipy, because a GPU die has
    a fast time constant on top of the heatsink's slow one.
Also reads where the GPU's own thermal slowdown starts (NVML clock event
reason bit 0x20, software thermal slowdown) as a measured throttle onset.

Usage: python experiments/thermal/calibrate_laptop.py <fit_run> [<validate_run> ...]
Writes results/thermal/laptop_rc.json (data_kind="measured").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from heatwave.data.schema import PROCESSED, load_series  # noqa: E402
from heatwave.provenance import write_result  # noqa: E402

from gpu_stack.research import heatwave_thermal as ht  # noqa: E402

SW_THERMAL = 0x20
HW_THERMAL = 0x40


def load(run: str):
    df, meta = load_series(f"laptop_{run}_1s")
    df = df.dropna(subset=["power_energy_w", "temp_c"])
    return df, meta


def two_node_sim(p, power, dt, t0_die, t0_sink):
    """Die (Cd, Rds to sink) + sink (Cs, Rsa to ambient Ta). Explicit substeps."""
    cd, rds, cs, rsa, ta = p
    n = len(power)
    td, ts = np.empty(n), np.empty(n)
    d, s = t0_die, t0_sink
    sub = max(1, int(np.ceil(dt / min(0.2 * cd * rds, 0.2 * cs * rsa, dt))))
    h = dt / sub
    for i in range(n):
        td[i], ts[i] = d, s
        for _ in range(sub):
            q = (d - s) / rds
            d += h * (power[i] - q) / cd
            s += h * (q - (s - ta) / rsa) / cs
    return td


def fit_two_node(power, temp, dt):
    def resid(lp):
        p = np.exp(lp[:4]).tolist() + [lp[4]]
        return two_node_sim(p, power, dt, temp[0], temp[0]) - temp

    best = None
    for cd0, cs0 in ((5.0, 300.0), (20.0, 1000.0), (2.0, 100.0)):
        x0 = np.array([np.log(cd0), np.log(0.3), np.log(cs0), np.log(0.4), 30.0])
        r = least_squares(resid, x0, method="trf", max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    p = np.exp(best.x[:4]).tolist() + [best.x[4]]
    return p


def score(pred, temp):
    err = pred - temp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1 - np.sum(err ** 2) / np.sum((temp - temp.mean()) ** 2))
    return {"rmse_c": rmse, "r2": r2, "max_abs_err_c": float(np.max(np.abs(err)))}


def throttle_onset(run: str):
    df, _ = load_series(f"laptop_{run}_10hz")
    r = df["clock_event_reasons"].fillna(0).astype(int)
    sw = (r & SW_THERMAL) > 0
    hw = (r & HW_THERMAL) > 0
    out = {"fraction_samples_sw_thermal": float(sw.mean()), "fraction_samples_hw_thermal": float(hw.mean())}
    if sw.any():
        out["temp_c_at_sw_thermal_p05"] = float(np.percentile(df.loc[sw, "temp_c"], 5))
        out["temp_c_at_sw_thermal_median"] = float(df.loc[sw, "temp_c"].median())
        out["temp_c_max"] = float(df["temp_c"].max())
        on = sw & (df["training_on"] > 0)
        off = (~sw) & (df["training_on"] > 0)
        out["sm_clock_mhz_training_sw_thermal_median"] = float(df.loc[on, "sm_clock_mhz"].median()) if on.any() else None
        out["sm_clock_mhz_training_no_thermal_median"] = float(df.loc[off, "sm_clock_mhz"].median()) if off.any() else None
    return out


def main(fit_run: str, val_runs: list[str]) -> None:
    dt = 1.0
    df, meta = load(fit_run)
    P, T = df["power_energy_w"].to_numpy(), df["temp_c"].to_numpy()
    amb = np.zeros_like(P)
    rc1, rep1 = ht.fit(P, amb, T, dt, initial={"heat_capacity_j_per_k": 30.0,
                                               "resistance_k_per_w": 0.5,
                                               "supply_setpoint_c": 30.0})
    p2 = fit_two_node(P, T, dt)
    res = {
        "fit_run": fit_run,
        "device": meta["source"]["device"],
        "one_node": {"params": rc1.to_dict(), "fit_report": rep1,
                     "in_sample": score(ht.simulate(rc1, P, amb, dt, T[0]), T)},
        "two_node": {"params": dict(zip(["die_heat_capacity_j_per_k", "die_to_sink_k_per_w",
                                          "sink_heat_capacity_j_per_k", "sink_to_ambient_k_per_w",
                                          "ambient_c"], p2)),
                     "die_tau_s": p2[0] * p2[1], "sink_tau_s": p2[2] * p2[3],
                     "in_sample": score(two_node_sim(p2, P, dt, T[0], T[0]), T)},
        "throttle": {fit_run: throttle_onset(fit_run)},
        "validation": {},
        "power_signal": "power_energy_w (1 s average from the NVML energy counter)",
    }
    inputs = [PROCESSED / f"laptop_{fit_run}_1s.parquet"]
    for v in val_runs:
        dv, _ = load(v)
        Pv, Tv = dv["power_energy_w"].to_numpy(), dv["temp_c"].to_numpy()
        res["validation"][v] = {
            "one_node": score(ht.simulate(rc1, Pv, np.zeros_like(Pv), dt, Tv[0]), Tv),
            "two_node": score(two_node_sim(p2, Pv, dt, Tv[0], Tv[0]), Tv),
            "n_seconds": int(len(Tv)),
        }
        res["throttle"][v] = throttle_onset(v)
        inputs.append(PROCESSED / f"laptop_{v}_1s.parquet")
    write_result(ROOT / "results/thermal/laptop_rc.json", res, script=__file__,
                 data_kind="measured", inputs=inputs)
    import json
    print(json.dumps({k: res[k] for k in ("one_node", "two_node", "validation", "throttle")}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
