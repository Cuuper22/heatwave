"""Warn before thermal throttling on MEASURED laptop runs, from temperature forecasts.

Event: a thermal slowdown onset = the first second of an episode in which
NVML reports the software thermal slowdown clock-event bit (0x20), preceded by
at least `gap_s` seconds without it. This is the GPU's own throttle signal,
not a modeled one.

Every `stride_s` seconds (origin t), each method forecasts GPU temperature for
the next `horizon_s` seconds from the last `context_s` seconds (1 s data):
  - timestral-<size> (zero-shot), without covariates and with the known-future
    load schedule (training_on, ckpt_scheduled) as covariates
  - ets, arima (classical, univariate)
  - naive-last (temperature stays where it is)
  - reactive: no forecast; alarm when the current temperature >= threshold
Scoring is per decision point, only at points where the GPU is not
throttling: does throttling start within (t + min_lead, t + horizon]?
A warning is raised when the forecast probability that temperature exceeds the
threshold T within the horizon is >= p. The probability is read off the 21
quantiles: 1 - (lowest level whose quantile path crosses T at any step).

The threshold T and probability p are tuned on the fit run (run1) per method
(maximize recall at >= `min_lead_s` lead minus false alarms), then frozen and
scored on the test run(s). Writes results/thermal/throttle_warning.json
(data_kind="measured").

Usage: python experiments/thermal/throttle_warning_laptop.py run1 run2 [--sizes 5M,28M]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from heatwave.data.schema import PROCESSED, load_series  # noqa: E402
from heatwave.forecast import ARIMAForecaster, ETSForecaster, QUANTILE_LEVELS  # noqa: E402
from heatwave.provenance import write_result  # noqa: E402

SW_THERMAL = 0x20
LEVELS = np.asarray(QUANTILE_LEVELS)


def load_run(run: str):
    d1, _ = load_series(f"laptop_{run}_1s")
    d10, _ = load_series(f"laptop_{run}_10hz")
    sw = ((d10["clock_event_reasons"].fillna(0).astype(int) & SW_THERMAL) > 0).astype(float)
    d1["sw_thermal"] = sw.resample("1s").max().reindex(d1.index).fillna(0.0)
    d1["temp_c"] = d1["temp_c"].interpolate(limit_direction="both")
    for c in ("training_on", "ckpt_scheduled"):
        d1[c] = d1[c].fillna(0.0)
    return d1


def onsets(sw: np.ndarray, gap_s: int) -> np.ndarray:
    on = []
    last_on = -10 ** 9
    for i in range(1, len(sw)):
        if sw[i] > 0 and sw[i - 1] == 0:
            if i - last_on >= gap_s and not sw[max(0, i - gap_s):i].any():
                on.append(i)
        if sw[i] > 0:
            last_on = i
    return np.asarray(on, dtype=int)


def prob_exceed(q: np.ndarray, thr: float) -> float:
    """q: (H, 21) quantile paths. P(max over horizon > thr) from the quantile grid."""
    crossing = (q > thr).any(axis=0)  # per level
    if not crossing.any():
        return 0.0
    return float(1.0 - LEVELS[np.argmax(crossing)])


def forecasts(method, temp, cov, origins, context_s, horizon_s):
    """Return (n_origins, H, 21) quantile paths for one method."""
    name, fc = method
    out = np.empty((len(origins), horizon_s, len(LEVELS)))
    if name == "naive-last":
        for j, t in enumerate(origins):
            out[j] = temp[t - 1]
        return out
    batch = 32
    for s in range(0, len(origins), batch):
        ob = origins[s:s + batch]
        ctx = np.stack([temp[t - context_s:t] for t in ob])[:, None, :]
        kw = {}
        if name.endswith("+schedule"):
            kw["known_covariates"] = np.stack([cov[:, t - context_s:t + horizon_s] for t in ob])
        out[s:s + len(ob)] = fc.forecast(ctx, horizon_s, **kw).values[:, 0]
    return out


def score(alarm: np.ndarray, origins: np.ndarray, sw: np.ndarray, ons: np.ndarray, horizon_s: int, min_lead_s: int):
    """Per-origin classification, only at origins where the GPU is not throttling.

    positive origin: throttling (bit 0x20) starts within (t + min_lead_s, t + horizon_s].
    Also per onset: lead time of the earliest alarm in [onset - horizon_s, onset - min_lead_s].
    """
    pos = np.array([sw[t + min_lead_s + 1:t + horizon_s + 1].any() for t in origins])
    tp = int((alarm & pos).sum())
    fp = int((alarm & ~pos).sum())
    fn = int((~alarm & pos).sum())
    leads, caught = [], 0
    for o in ons:
        win = (origins >= o - horizon_s) & (origins <= o - min_lead_s) & alarm
        if win.any():
            caught += 1
            leads.append(int(o - origins[np.argmax(win)]))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return {"origins": int(len(origins)), "positive_origins": int(pos.sum()),
            "precision": prec, "recall": rec, "f1": f1,
            "false_alarm_rate": fp / int((~pos).sum()) if (~pos).any() else None,
            "onsets": int(len(ons)), "onsets_warned": caught,
            "onset_recall": caught / len(ons) if len(ons) else None,
            "median_lead_s": float(np.median(leads)) if leads else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_run")
    ap.add_argument("test_runs", nargs="+")
    ap.add_argument("--sizes", default="5M,28M")
    ap.add_argument("--context-s", type=int, default=600)
    ap.add_argument("--horizon-s", type=int, default=30)
    ap.add_argument("--stride-s", type=int, default=2)
    ap.add_argument("--gap-s", type=int, default=30)
    ap.add_argument("--min-lead-s", type=int, default=3)
    ap.add_argument("--classical", default="ets,arima", help="comma list from ets,arima ('' for none)")
    a = ap.parse_args()
    from heatwave.forecast.timestral_fc import TimestralForecaster

    methods = [("naive-last", None)]
    cl = [c for c in a.classical.split(",") if c]
    if "ets" in cl:
        methods.append(("ets", ETSForecaster(1, max_context=a.context_s)))
    if "arima" in cl:
        methods.append(("arima", ARIMAForecaster((2, 1, 1), max_context=a.context_s)))
    for sz in a.sizes.split(","):
        tf = TimestralForecaster(sz, threads=2)
        methods += [(f"timestral-{sz}", tf), (f"timestral-{sz}+schedule", tf)]

    thr_grid = np.arange(80.0, 89.0, 1.0)
    p_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    runs = [a.fit_run] + a.test_runs
    data, cache = {}, {}
    for r in runs:
        d = load_run(r)
        temp = d["temp_c"].to_numpy()
        cov = d[["training_on", "ckpt_scheduled"]].to_numpy().T
        sw = d["sw_thermal"].to_numpy()
        origins = np.arange(a.context_s, len(temp) - a.horizon_s, a.stride_s)
        # only origins where the GPU is not throttling now (and was not in the last 5 s)
        origins = np.array([t for t in origins if not sw[max(0, t - 5):t + 1].any()], dtype=int)
        ons = onsets(sw, a.gap_s)
        data[r] = (temp, cov, origins, ons, sw)
        for m in methods:
            cache[(r, m[0])] = forecasts(m, temp, cov, origins, a.context_s, a.horizon_s)
        print(r, "seconds", len(temp), "onsets", len(ons), "origins", len(origins), flush=True)

    def alarms(r, name, thr, p):
        temp, _, origins, _, _ = data[r]
        if name == "reactive":
            return temp[origins - 1] >= thr
        q = cache[(r, name)]
        return np.array([prob_exceed(q[j], thr) >= p for j in range(len(origins))])

    names = ["reactive"] + [m[0] for m in methods]
    out = {"settings": vars(a), "event": "NVML software thermal slowdown (bit 0x20) onset",
           "runs": {r: {"seconds": int(len(data[r][0])), "onsets": int(len(data[r][3]))} for r in runs},
           "methods": {}}
    for name in names:
        best, best_val = None, -1e9
        for thr in thr_grid:
            for p in ([1.0] if name == "reactive" else p_grid):
                _, _, origins, ons, sw = data[a.fit_run]
                s = score(alarms(a.fit_run, name, thr, p), origins, sw, ons, a.horizon_s, a.min_lead_s)
                val = s["f1"]
                if val > best_val:
                    best, best_val = (float(thr), float(p), s), val
        thr, p, fit_s = best
        res = {"tuned_on": a.fit_run, "threshold_c": thr, "probability": p, "fit": fit_s, "test": {}}
        for r in a.test_runs:
            _, _, origins, ons, sw = data[r]
            res["test"][r] = score(alarms(r, name, thr, p), origins, sw, ons, a.horizon_s, a.min_lead_s)
        out["methods"][name] = res
        print(name, "thr", thr, "p", p, "test", res["test"], flush=True)
    write_result(ROOT / "results/thermal/throttle_warning.json", out, script=__file__,
                 data_kind="measured", inputs=[PROCESSED / f"laptop_{r}_1s.parquet" for r in runs])


if __name__ == "__main__":
    main()
