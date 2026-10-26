"""Multi-scale rolling-origin forecasting evaluation (seconds, minutes, hours).

Forecasters (all univariate, no covariates; covariate ablation is task T04b): Timestral 5M and 28M (zero-shot) on every
window, Timestral 209M zero-shot on an equally spaced subsample of windows per dataset (``--n209``), SeasonalNaive, ETS
(AutoETS, unrestricted and restricted to no trend: ets / ets-flat) and ARIMA(2,1,1) (seasonal (1,0,0) at hourly scale) from the shared package. A fine-tuned checkpoint can be added
with ``--finetuned-dir DIR --finetuned-size 28M`` (scored as ``timestral-<size>-finetuned``).

Protocol (fixed before running; nothing is tuned on test windows):
  seconds: context 256 steps of 1 s, horizon 300 s; minutes: context 96 steps, horizon 60 steps (1 min; 15 s for the laptop);
  hours: context 96 h, horizon 48 h. Origins are equally spaced (stride and per-series cap in ``datasets.dataset_specs``),
  taken from the held-out parts defined in ``heatwave/forecast/datasets.py``. Metrics: WQL, MASE of the median, MAE, 80% and
  98% coverage, per horizon bucket, with 95% bootstrap CIs over windows (origins) and over groups.

Usage: python experiments/forecast/multiscale.py [--scales seconds,minutes,hours] [--datasets a,b] [--n209 120]
       [--threads 2] [--workers 1] [--sim-seeds 100,101,...] [--max-windows N] (development subsample) [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ALL_METHODS, N_BOOT, bootstrap_summary, forecast_series, hardware_note, log, make_timestral,  # noqa: E402
                     score_windows)

from heatwave.forecast import datasets as D  # noqa: E402
from heatwave.forecast.datasets import CONTEXT, HORIZON, BUCKETS, SEASON  # noqa: E402
from heatwave.provenance import write_result  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="seconds,minutes,hours")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--n209", type=int, default=120, help="windows per dataset for Timestral 209M (equally spaced subsample)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1, help="processes for the classical baselines")
    ap.add_argument("--sim-seeds", default=",".join(str(s) for s in D.SIM_TEST_SEEDS))
    ap.add_argument("--max-windows", type=int, default=0, help="development cap per dataset (0 = no cap); recorded in the result")
    ap.add_argument("--finetuned-dir", default="")
    ap.add_argument("--finetuned-size", default="28M")
    ap.add_argument("--platform-label", default="")
    ap.add_argument("--no-mit", action="store_true", help="skip the MIT Supercloud datasets (local copy only)")
    ap.add_argument("--out", default=str(ROOT / "results" / "forecast" / "multiscale.json"))
    return ap.parse_args()


def main() -> None:
    a = parse()
    t_start = time.time()
    seeds = tuple(int(s) for s in a.sim_seeds.split(",") if s)
    specs = D.dataset_specs(seeds, mit=not a.no_mit)
    scales = set(a.scales.split(","))
    want = set(x for x in a.datasets.split(",") if x)
    methods = [m for m in a.methods.split(",") if m]
    ft = f"timestral-{a.finetuned_size}-finetuned" if a.finetuned_dir else None
    result: dict = {
        "protocol": {
            "context_steps": CONTEXT, "horizon_steps": HORIZON, "season_steps": SEASON,
            "buckets": {k: [list(b) for b in v] for k, v in BUCKETS.items()},
            "metrics": "WQL = 2*mean pinball(21 levels)/mean|y| (ratio of window means); MASE = mean over windows of MAE(median)/in-context seasonal-naive MAE; "
                       "MAE of median; coverage of the 10-90% (cov80) and 1-99% (cov98) quantile intervals",
            "bootstrap": {"n": N_BOOT, "ci_origin": "windows resampled with replacement", "ci_group": "groups (session/run/job/seed) resampled with replacement"},
            "classical": {"seasonal-naive": "season = SEASON[scale], quantiles from in-sample seasonal-naive errors",
                          "ets": "statsforecast AutoETS (shared ETSForecaster, unrestricted model choice), season = SEASON[scale]",
                          "ets-flat": "statsforecast AutoETS restricted to additive error, no trend (ANN; ANA at hourly scale)",
                          "arima": "statsforecast ARIMA(2,1,1); hours: seasonal (1,0,0) with season 24"},
            "timestral": "zero-shot, no covariates, V=1 per series; 209M on an equally spaced subsample of windows",
            "n209_windows_per_dataset": a.n209,
            "covariates_used": False,
            "weights_dir": a.finetuned_dir or None,
        },
        "datasets": {},
        "hardware": a.platform_label or hardware_note(),
        "threads": a.threads, "workers": a.workers,
        "development_cap_per_dataset": a.max_windows or None,
        "sim_test_seeds": list(seeds),
    }
    inputs: list[Path] = []
    ts_cache: dict = {}
    for name, spec in specs.items():
        if spec.scale not in scales or (want and name not in want):
            continue
        t_d = time.time()
        try:
            w = D.build_dataset(spec)
        except FileNotFoundError as e:  # e.g. simulator cache missing
            log(name, "skipped:", e)
            result["datasets"][name] = {"skipped": str(e), "data_kind": spec.data_kind}
            continue
        if a.max_windows and w.n > a.max_windows:
            w = D.subsample_windows(w, a.max_windows)
        log(f"{name}: {w.n} windows ({w.meta['n_series']} series, {w.data_kind}), context {w.context.shape[1]}, horizon {w.future.shape[1]}")
        sub_idx = D.subsample_index(w.n, a.n209)
        H = w.future.shape[1]
        buckets = BUCKETS[w.scale]
        block: dict = {"scale": w.scale, "data_kind": w.data_kind, "step_s": w.step_s, "season_steps": w.season, "meta": w.meta,
                       "unit": spec.unit, "methods": {}, "subset_209M": {"indices_n": int(len(sub_idx)), "of": int(w.n), "methods": {}}}
        comps: dict[str, dict] = {}
        for m in methods + ([ft] if ft else []):
            on_subset = m == "timestral-209M"
            ctx = w.context[sub_idx] if on_subset else w.context
            key = f"ms__{name}__n{w.n}"
            try:
                if m == ft:
                    vals, sec, cached = forecast_series(f"timestral-{a.finetuned_size}", w.scale, ctx, H, threads=a.threads,
                                                        weights_dir=a.finetuned_dir, cache_key=key + "__ft")
                else:
                    ts = None
                    if m.startswith("timestral"):
                        ts = ts_cache.get(m) or ts_cache.setdefault(m, make_timestral(m, a.threads))
                    vals, sec, cached = forecast_series(m, w.scale, ctx, H, threads=a.threads, workers=a.workers,
                                                        cache_key=key + ("__sub" if on_subset else ""), timestral=ts)
            except Exception as e:  # record, keep going
                log(f"  {m}: FAILED {type(e).__name__}: {e}")
                block["methods"][m] = {"error": f"{type(e).__name__}: {e}"}
                continue
            fut = w.future[sub_idx] if on_subset else w.future
            cx = w.context[sub_idx] if on_subset else w.context
            grp = w.group[sub_idx] if on_subset else w.group
            comp = score_windows(vals, fut, cx, w.season, buckets)
            comps[m] = {"vals": vals if not on_subset else None, "comp": comp, "subset": on_subset, "group": grp}
            entry = {"seconds_per_series": sec, "from_cache": cached, "n_windows": int(vals.shape[0]),
                     "buckets": {b: bootstrap_summary(c, grp) for b, c in comp.items()}}
            (block["subset_209M"]["methods"] if on_subset else block["methods"])[m] = entry
            log(f"  {m}: {sec * 1000:.1f} ms/series; WQL(all) {entry['buckets']['all']['wql']['est']:.4f}")
        # every other method scored on the same subset as 209M
        if "timestral-209M" in comps:
            for m, c in comps.items():
                if c["subset"]:
                    continue
                sub = {b: {k: v[sub_idx] for k, v in cc.items()} for b, cc in c["comp"].items()}
                block["subset_209M"]["methods"][m] = {"n_windows": int(len(sub_idx)),
                                                     "buckets": {b: bootstrap_summary(cc, w.group[sub_idx]) for b, cc in sub.items()}}
        result["datasets"][name] = block
        block["seconds_dataset"] = time.time() - t_d
    result["runtime_seconds"] = time.time() - t_start
    for p in ("laptop_run2_1s", "public_caiso_np15_lmp_rtm", "public_uk_carbon_intensity"):
        f = ROOT / "data" / "processed" / f"{p}.parquet"
        if f.exists():
            inputs.append(f)
    inputs += [D.sim_cache_path(s) for s in seeds if D.sim_cache_path(s).exists()]
    notes = ("Datasets of kind 'simulated' are gpu_stack heatwave simulator v2 output (not measured). "
             f"Hardware: {result['hardware']}. Timestral zero-shot weights from the Hugging Face hub.")
    write_result(a.out, result, script=__file__, data_kind="mixed", inputs=inputs, notes=notes)
    log("wrote", a.out, f"({result['runtime_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
