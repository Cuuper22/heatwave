"""Hierarchical forecast coherence-adjustment study: GPU -> rack -> site (SIMULATED) and GPU -> node (public).

Every node of the hierarchy (each GPU, each rack, each site / node) is forecast independently (univariate, V=1) from its
own 256-step context, horizon 300 s, 1 s data. The base forecasts are then adjusted for coherence in ``heatwave/forecast/hier_coherence.py`` (also importable under the task-file name):

  bottom_up         G = [0 | I]
  ols               G = (S'S)^-1 S'
  mint_shrink       MinT, W = Schaefer-Strimmer shrinkage covariance of the base MEDIAN errors at horizon steps 1-10
                    on TRAINING windows (public: training sessions; simulated: validation seeds 50-53 in four calibration windows at times disjoint from the test windows), as the
                    MinT paper uses one-step residuals for every horizon
  mint_shrink_allh  the same with errors pooled over all 300 steps (sensitivity)

The adjustment is applied to each of the 21 quantile levels separately with the same G (an approximation: coherent number
vectors, not the quantiles of a coherent joint distribution) and to the median. Reported: coherence error before and after
(relative L1 distance of each aggregate from the sum of the bottom series below it; also after the sorting that removes
quantile crossings), and WQL / MASE / coverage per level and horizon bucket before and after, with paired bootstrap
differences to the base forecasts.

Usage: python experiments/forecast/hierarchy.py [--studies public,sim] [--methods ...] [--sim-seeds 100,...]
       [--stride-public 60] [--stride-sim 500] [--max-test-windows N] [--threads 2] [--workers 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (N_BOOT, bootstrap_delta, bootstrap_summary, ctx_hash, forecast_series, hardware_note, log,  # noqa: E402
                     make_timestral, score_block)

from heatwave.forecast import MEDIAN_INDEX  # noqa: E402
from heatwave.forecast import datasets as D  # noqa: E402
from heatwave.forecast import hier_coherence as R  # noqa: E402
from heatwave.forecast.datasets import BUCKETS, HierWindows  # noqa: E402
from heatwave.provenance import write_result  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("timestral-5M", "timestral-28M", "seasonal-naive", "ets", "arima")
W_STEPS = 10
COMP = ("pin", "absy", "mae", "mase", "cov80", "cov98")


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", default="public,sim")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--sim-seeds", default=",".join(str(s) for s in D.SIM_TEST_SEEDS))
    ap.add_argument("--val-seeds", default=",".join(str(s) for s in D.SIM_VAL_SEEDS))
    ap.add_argument("--stride-public", type=int, default=60)
    ap.add_argument("--stride-sim", type=int, default=500)
    ap.add_argument("--max-test-windows", type=int, default=0, help="development cap per tree (0 = none); recorded")
    ap.add_argument("--max-train-windows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--finetuned-dir", default="", help="fine-tuned Timestral checkpoint directory (weights_dir); adds timestral-<size>-finetuned")
    ap.add_argument("--finetuned-size", default="28M")
    ap.add_argument("--platform-label", default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "forecast" / "hierarchy.json"))
    return ap.parse_args()


def cap(hw: HierWindows, k: int) -> HierWindows:
    if not k or hw.n <= k:
        return hw
    idx = D.subsample_index(hw.n, k)
    from dataclasses import replace

    return replace(hw, context=hw.context[idx], future=hw.future[idx], window_id=hw.window_id[idx], group=hw.group[idx],
                   meta={**hw.meta, "subsample_of": hw.n, "subsample_n": int(len(idx))})


def base_forecasts(method: str, hw: HierWindows, key: str, a, ts) -> tuple[np.ndarray, float]:
    N, n, T = hw.context.shape
    H = hw.future.shape[2]
    ft = method.endswith("-finetuned")
    vals, sec, _ = forecast_series(method.replace("-finetuned", ""), "seconds", hw.context.reshape(N * n, T), H, threads=a.threads,
                                   workers=a.workers, cache_key=f"hier__{key}__n{N}" + ("__ft" if ft else ""),
                                   timestral=None if ft else ts, weights_dir=a.finetuned_dir if ft else None)
    return vals.reshape(N, n, H, vals.shape[-1]), sec


def residual_matrix(vals: np.ndarray, fut: np.ndarray, steps: int | None) -> np.ndarray:
    med = vals[..., MEDIAN_INDEX].astype(np.float64)
    e = med - fut  # (N, n, H)
    if steps:
        e = e[:, :, :steps]
    return e.transpose(0, 2, 1).reshape(-1, e.shape[1])


def evaluate_variant(P: np.ndarray | None, vals: np.ndarray, hw: HierWindows, chunk: int = 20) -> dict:
    """Adjust (P=None: base) and score by level and horizon bucket; returns per-window components and coherence sums."""
    hier = hw.hierarchy
    N = vals.shape[0]
    H = hw.future.shape[2]
    buckets = list(BUCKETS["seconds"]) + [("all", 1, H)]
    comps = {lv: {b[0]: {k: np.empty(N) for k in COMP} for b in buckets} for lv in hier.level_order}
    acc = {"raw": {}, "sorted": {}, "children": {}, "median": {}}
    n_cross = 0
    n_tot = 0
    n_neg = 0
    n_bot = 0
    for s0 in range(0, N, chunk):
        sl = slice(s0, min(N, s0 + chunk))
        v = vals[sl].astype(np.float64)
        r = v if P is None else R.adjust(v, P)
        srt = np.sort(r, axis=-1)
        n_cross += int((np.diff(r, axis=-1) < 0).any(axis=-1).sum())
        n_tot += int(np.prod(r.shape[:3]))
        n_neg += int((srt[:, hier.bottom_index, :, MEDIAN_INDEX] < 0).sum())
        n_bot += int(srt[:, hier.bottom_index, :, MEDIAN_INDEX].size)
        for name, arr, vs in (("raw", r, "bottom"), ("sorted", srt, "bottom"), ("children", srt, "children"),
                              ("median", srt[..., MEDIAN_INDEX:MEDIAN_INDEX + 1], "bottom")):
            for lv, (num, den) in R.coherence_sums(arr, hier, vs).items():
                o = acc[name].setdefault(lv, [0.0, 0.0])
                o[0] = o[0] + num
                o[1] = o[1] + den
        for i in range(sl.start, sl.stop):
            for lv in hier.level_order:
                idx = hier.level_index(lv)
                for label, lo, hi in buckets:
                    res = score_block(srt[i - s0, idx][:, None], hw.future[i, idx], hw.context[i, idx], hw.season, lo, min(hi, H))
                    for k in COMP:
                        comps[lv][label][k][i] = res[k]
    coh = {}
    for name, d in acc.items():
        coh[name] = {}
        for lv, (num, den) in d.items():
            num, den = np.atleast_1d(num), np.atleast_1d(den)
            tot = float(num.sum() / den.sum()) if den.sum() > 0 else None
            coh[name][lv] = {"pooled": tot, "per_quantile": (num / den).tolist() if len(num) > 1 else None}
    return {"comps": comps, "coherence": coh, "quantile_crossing_share": n_cross / n_tot if n_tot else None,
            "negative_bottom_median_share": n_neg / n_bot if n_bot else None}


def run_tree(name: str, test: HierWindows, train: HierWindows, methods, a, ts_cache) -> dict:
    hier = test.hierarchy
    S = hier.S
    out = {"data_kind": test.data_kind, "hierarchy": {"names_head": list(hier.names[:6]), "n_nodes": hier.n, "n_bottom": hier.m,
                                                      "levels": {lv: int(len(hier.level_index(lv))) for lv in hier.level_order}},
           "test_meta": test.meta, "train_meta": train.meta, "n_test_windows": test.n, "n_train_windows": train.n,
           "w_steps_primary": W_STEPS, "methods": {}}
    # data coherence check: actuals add up
    fut_coh = R.coherence_error(test.future[..., None], hier)
    out["actuals_coherence_error"] = fut_coh
    for m in methods:
        t0 = time.time()
        try:
            ts = None
            if m.startswith("timestral") and not m.endswith("-finetuned"):
                ts = ts_cache.get(m) or ts_cache.setdefault(m, make_timestral(m, a.threads))
            tr_vals, sec_tr = base_forecasts(m, train, f"{name}__train", a, ts)
            te_vals, sec_te = base_forecasts(m, test, f"{name}__test", a, ts)
        except Exception as e:
            log(f"  {name}/{m}: FAILED {type(e).__name__}: {e}")
            out["methods"][m] = {"error": f"{type(e).__name__}: {e}"}
            continue
        W1, lam1 = R.shrink_covariance(residual_matrix(tr_vals, train.future, W_STEPS))
        Wa, lama = R.shrink_covariance(residual_matrix(tr_vals, train.future, None))
        variants = {
            "base": None,
            "bottom_up": R.projection(S, R.g_bottom_up(S)),
            "ols": R.projection(S, R.g_ols(S)),
            "mint_shrink": R.projection(S, R.g_mint(S, W1)),
            "mint_shrink_allh": R.projection(S, R.g_mint(S, Wa)),
        }
        res = {"seconds_per_series_test": sec_te, "mint_shrinkage_lambda": {"steps_1_10": lam1, "all_steps": lama}, "variants": {}}
        evals = {v: evaluate_variant(P, te_vals, test) for v, P in variants.items()}
        for v, ev in evals.items():
            entry = {"coherence_error": ev["coherence"], "quantile_crossing_share_before_sort": ev["quantile_crossing_share"],
                     "negative_bottom_median_share": ev["negative_bottom_median_share"], "levels": {}}
            for lv in hier.level_order:
                entry["levels"][lv] = {"n_nodes": int(len(hier.level_index(lv))),
                                       "buckets": {b: bootstrap_summary(c, test.group) for b, c in ev["comps"][lv].items()}}
                if v != "base":
                    entry["levels"][lv]["delta_vs_base"] = {
                        b: bootstrap_delta(c, evals["base"]["comps"][lv][b], test.group) for b, c in ev["comps"][lv].items()}
            res["variants"][v] = entry
        out["methods"][m] = res
        w = res["variants"]
        log(f"  {name}/{m}: {time.time() - t0:.0f} s; lambda {lam1:.3f}; coherence(base->mint) "
            + ", ".join(f"{lv}: {w['base']['coherence_error']['sorted'][lv]['pooled']:.3f}->{w['mint_shrink']['coherence_error']['sorted'][lv]['pooled']:.1e}"
                        for lv in hier.level_order[:-1]))
    return out


def main() -> None:
    a = parse()
    t_start = time.time()
    methods = [m for m in a.methods.split(",") if m]
    if a.finetuned_dir:
        methods.append(f"timestral-{a.finetuned_size}-finetuned")
    studies = set(a.studies.split(","))
    seeds = tuple(int(s) for s in a.sim_seeds.split(",") if s)
    vseeds = tuple(int(s) for s in a.val_seeds.split(",") if s)
    ts_cache: dict = {}
    result: dict = {
        "protocol": {
            "context_steps": D.CONTEXT["seconds"], "horizon_steps": D.HORIZON["seconds"], "step_s": 1.0,
            "buckets": [list(b) for b in BUCKETS["seconds"]],
            "variants": ["base", "bottom_up", "ols", "mint_shrink", "mint_shrink_allh"],
            "mint_covariance": "Schaefer-Strimmer shrinkage of base-median errors on training windows (steps 1-10, or all steps for mint_shrink_allh); simulated trees: validation seeds, calibration windows disjoint in time from the test windows",
            "quantile_adjustment": "each of the 21 quantile levels adjusted separately with the same G (approximation); median = level 0.5",
            "coherence_error": "sum |y_A - sum of bottom series below A| / sum |y_A| per aggregate level, pooled over windows, steps, nodes; quantile entries per level",
            "metrics": "as multiscale.json; level-wise WQL = 2*mean pinball/mean|y| over the nodes of the level; MASE with in-context naive (lag 1) scale",
            "bootstrap": {"n": N_BOOT, "unit": "windows (origins); group = session or sim seed"},
            "covariates_used": False, "classical": "seasonal-naive (season 1 = last value), AutoETS, ARIMA(2,1,1)",
            "timestral": "zero-shot 5M and 28M; 209M not run in this study",
        },
        "trees": {}, "hardware": a.platform_label or hardware_note(), "threads": a.threads, "workers": a.workers,
        "sim_test_seeds": list(seeds), "sim_val_seeds_for_mint_covariance": list(vseeds),
    }
    inputs: list[Path] = []
    if "public" in studies:
        log("public GPU -> node")
        te = cap(D.hier_public_node("test", a.stride_public), a.max_test_windows)
        tr = cap(D.hier_public_node("train", a.stride_public), a.max_train_windows)
        log(f"  test windows {te.n}, train windows {tr.n}")
        result["trees"]["public_gpu_node"] = run_tree("public_gpu_node", te, tr, methods, a, ts_cache)
    if "sim" in studies:
        log("SIMULATED GPU -> rack -> site")
        tes = D.hier_sim("test", seeds, a.stride_sim)
        trs = D.hier_sim("val", vseeds, a.stride_sim)
        for sid in tes:
            te = cap(tes[sid], a.max_test_windows)
            tr = cap(trs[sid], a.max_train_windows)
            log(f"  {sid}: test windows {te.n}, validation windows {tr.n}, nodes {te.hierarchy.n}")
            result["trees"][f"sim_{sid}"] = run_tree(f"sim_{sid}", te, tr, methods, a, ts_cache)
        inputs += [D.sim_cache_path(s) for s in list(seeds) + list(vseeds) if D.sim_cache_path(s).exists()]
    result["runtime_seconds"] = time.time() - t_start
    result["development_cap_test_windows"] = a.max_test_windows or None
    kinds = {t["data_kind"] for t in result["trees"].values()}
    kind = "mixed" if len(kinds) > 1 else (kinds.pop() if kinds else "mixed")
    notes = (f"Trees with data_kind 'simulated' are gpu_stack heatwave simulator v2 output. Hardware: {result['hardware']}. "
             "Quantile levels are adjusted one at a time (approximation).")
    write_result(a.out, result, script=__file__, data_kind=kind, inputs=inputs, notes=notes)
    log("wrote", a.out, f"({result['runtime_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
