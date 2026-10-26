"""Shared evaluation engine for the covariate ablations.

A *dataset* yields batches of forecast windows (context, realised future, one array per covariate group). The engine
runs every (method, covariate configuration) on every window, stores per-window metric statistics, and summarises
them per horizon bucket with paired block-bootstrap confidence intervals over forecast origins.

Per-window statistics (per horizon bucket and for the whole horizon), columns of the last axis of `stats`:
  0 pinball   mean pinball loss over the 21 quantile levels and the bucket's steps   (heatwave.forecast.metrics.pinball)
  1 abs_y     mean |actual|                                                          (denominator of WQL)
  2 mae       mean |median - actual|                                                  (numerator of MASE)
  3 mase      mae / mean |context[s:] - context[:-s]| with s = scale.season           (heatwave.forecast.metrics.mase)
  4 cov80     fraction of steps inside the central 80% interval                      (metrics.coverage, level 0.8)
  5 cov98     fraction inside the central 98% interval                               (metrics.coverage, level 0.98)

Aggregates: WQL = 2 * mean(pinball) / mean(abs_y) over windows (equals metrics.wql on the pooled windows when every
window has the same number of steps), MASE = mean of per-window MASE, coverage = mean over windows.
tests/test_covariates.py checks the per-window statistics against heatwave.forecast.metrics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from heatwave.forecast import QUANTILE_LEVELS, QuantileForecast

LEVELS = np.asarray(QUANTILE_LEVELS)
STAT_NAMES = ("pinball", "abs_y", "mae", "mase", "cov80", "cov98")
METRIC_NAMES = ("wql", "mase", "cov80", "cov98")
_I80 = (QUANTILE_LEVELS.index(0.1), QUANTILE_LEVELS.index(0.9))
_I98 = (QUANTILE_LEVELS.index(0.01), QUANTILE_LEVELS.index(0.99))


@dataclass(frozen=True)
class Scale:
    """Fixed evaluation geometry of one dataset (declared before any score was computed)."""

    name: str
    context: int
    horizon: int
    buckets: tuple  # ((lo, hi), ...) 1-based inclusive horizon steps
    season: int  # MASE scale lag and the seasonal-naive reference period, in steps
    step: str  # human label of one step, e.g. "1 min"
    bucket_labels: tuple = ()

    def labels(self) -> list[str]:
        if self.bucket_labels:
            return list(self.bucket_labels) + ["all"]
        return [f"{lo}-{hi}" for lo, hi in self.buckets] + ["all"]

    def edges(self) -> list[tuple[int, int]]:
        return list(self.buckets) + [(1, self.horizon)]


@dataclass
class Batch:
    stream: np.ndarray  # (B,) stream index
    origin: np.ndarray  # (B,) position of the origin in the dataset's origin list
    ctx: np.ndarray  # (B, T)
    y: np.ndarray  # (B, H)
    cov: dict  # group -> (B, C_g, T + H)


def window_stats(q: np.ndarray, y: np.ndarray, ctx: np.ndarray, scale: Scale) -> np.ndarray:
    """q (B, H, 21), y (B, H), ctx (B, T) -> (B, n_buckets + 1, 6)."""
    d = y[..., None] - q
    loss = np.maximum(LEVELS * d, (LEVELS - 1.0) * d).mean(axis=-1)  # (B, H), NaN where y is NaN
    med = q[..., QUANTILE_LEVELS.index(0.5)]
    ae = np.abs(med - y)
    in80 = ((y >= q[..., _I80[0]]) & (y <= q[..., _I80[1]])).astype(float)
    in98 = ((y >= q[..., _I98[0]]) & (y <= q[..., _I98[1]])).astype(float)
    nan = np.isnan(y)
    in80[nan] = np.nan
    in98[nan] = np.nan
    s = scale.season
    with np.errstate(all="ignore"):
        denom = np.nanmean(np.abs(ctx[:, s:] - ctx[:, :-s]), axis=1)
    denom = np.where(denom > 0, denom, np.nan)
    out = np.full((q.shape[0], len(scale.buckets) + 1, 6), np.nan)
    with np.errstate(all="ignore"):
        for j, (lo, hi) in enumerate(scale.edges()):
            sl = slice(lo - 1, hi)
            out[:, j, 0] = np.nanmean(loss[:, sl], axis=1)
            out[:, j, 1] = np.nanmean(np.abs(y[:, sl]), axis=1)
            out[:, j, 2] = np.nanmean(ae[:, sl], axis=1)
            out[:, j, 3] = out[:, j, 2] / denom
            out[:, j, 4] = np.nanmean(in80[:, sl], axis=1)
            out[:, j, 5] = np.nanmean(in98[:, sl], axis=1)
    out[np.isnan(q).any(axis=(1, 2))] = np.nan  # windows the model rejected
    return out


def sums_by_origin(stats: np.ndarray):
    """stats (S, O, nb, 6) -> per-origin sums over streams and valid counts, both (O, nb, 6)."""
    ok = ~np.isnan(stats)
    return np.where(ok, stats, 0.0).sum(axis=0), ok.sum(axis=0).astype(float)


def metrics_from_sums(s: np.ndarray, n: np.ndarray) -> np.ndarray:
    """s, n (..., nb, 6) summed over origins -> (..., nb, 4): wql, mase, cov80, cov98."""
    with np.errstate(all="ignore"):
        pin = s[..., 0] / n[..., 0]
        ay = s[..., 1] / n[..., 1]
        out = np.stack(
            [2.0 * pin / ay, s[..., 3] / n[..., 3], s[..., 4] / n[..., 4], s[..., 5] / n[..., 5]], axis=-1
        )
    return out


def block_indices(rng: np.random.Generator, n_origins: int, block: int, reps: int) -> np.ndarray:
    """Moving-block bootstrap indices (reps, n_origins) over the origin axis."""
    block = max(1, min(block, n_origins))
    nblocks = math.ceil(n_origins / block)
    starts = rng.integers(0, n_origins - block + 1, size=(reps, nblocks))
    idx = (starts[..., None] + np.arange(block)).reshape(reps, -1)
    return idx[:, :n_origins]


def summarize(
    stats_by_cfg: dict,
    *,
    none_key: str,
    block: int,
    cluster: Optional[np.ndarray] = None,
    reps: int = 1000,
    seed: int = 0,
    chunk: int = 50,
) -> dict:
    """Point estimates and paired bootstrap CIs for every config; deltas against ``none_key``.

    ``stats_by_cfg[cfg]`` is (S, O, nb, 6). Origin bootstrap: moving blocks of ``block`` consecutive origins, the same
    resampled origins for every stream and every config (paired). ``cluster`` (S,) optionally gives a cluster id per
    stream (the simulator seed); the cluster bootstrap resamples whole clusters with replacement.
    """
    cfgs = list(stats_by_cfg)
    S0 = stats_by_cfg[cfgs[0]]
    n_origins, nb = S0.shape[1], S0.shape[2]
    sums = {c: sums_by_origin(stats_by_cfg[c]) for c in cfgs}
    point = {c: metrics_from_sums(sums[c][0].sum(0), sums[c][1].sum(0)) for c in cfgs}
    rng = np.random.default_rng(seed)
    boot = {c: [] for c in cfgs}
    for r0 in range(0, reps, chunk):
        idx = block_indices(rng, n_origins, block, min(chunk, reps - r0))
        for c in cfgs:
            s, n = sums[c]
            boot[c].append(metrics_from_sums(s[idx].sum(1), n[idx].sum(1)))
    boot = {c: np.concatenate(v, axis=0) for c, v in boot.items()}  # (R, nb, 4)
    out = {"point": point, "boot": boot}
    if cluster is not None:
        ids = np.unique(cluster)
        csum = {}
        for c in cfgs:
            st = stats_by_cfg[c]
            ok = ~np.isnan(st)
            v = np.where(ok, st, 0.0)
            csum[c] = (
                np.stack([v[cluster == i].sum(axis=(0, 1)) for i in ids]),
                np.stack([ok[cluster == i].sum(axis=(0, 1)).astype(float) for i in ids]),
            )
        idx = rng.integers(0, len(ids), size=(reps, len(ids)))
        out["boot_cluster"] = {
            c: metrics_from_sums(csum[c][0][idx].sum(1), csum[c][1][idx].sum(1)) for c in cfgs
        }
    return out


def ci(a: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.nanpercentile(a, [2.5, 97.5], axis=axis)


@dataclass
class MethodSpec:
    name: str
    forecaster: object
    configs: Sequence  # list of (cfg_name, tuple_of_groups)
    uses_covariates: bool = True


def _forecast_batch(forecaster, ctx3, horizon, kw, fail_count: dict, cfg: str) -> np.ndarray:
    """Quantiles (B, H, 21). If the model rejects a batch (degenerate input), retry window by window; a window that still
    fails becomes NaN (excluded from every average) and is counted in ``fail_count[cfg]``."""
    try:
        return forecaster.forecast(ctx3, horizon, **kw).values[:, 0]
    except ValueError:
        out = np.full((ctx3.shape[0], horizon, len(QUANTILE_LEVELS)), np.nan)
        for b in range(ctx3.shape[0]):
            try:
                one = {k: v[b:b + 1] for k, v in kw.items()}
                out[b] = forecaster.forecast(ctx3[b:b + 1], horizon, **one).values[0, 0]
            except ValueError:
                fail_count[cfg] = fail_count.get(cfg, 0) + 1
        return out


def evaluate(
    dataset,
    methods: Sequence[MethodSpec],
    *,
    batch_size: int = 64,
    log: Callable[[str], None] = print,
) -> dict:
    """Run every method/config on every window. Returns {method: {cfg: stats (S, O, nb+1, 6)}}."""
    sc: Scale = dataset.scale
    S, O = dataset.n_streams, dataset.n_origins
    nb = len(sc.buckets) + 1
    stats = {m.name: {c: np.full((S, O, nb, 6), np.nan) for c, _ in m.configs} for m in methods}
    timing = {m.name: 0.0 for m in methods}
    failures = {m.name: {} for m in methods}
    t_start = time.time()
    done = 0
    total = S * O
    for batch in dataset.iter_batches(batch_size):
        ctx3 = batch.ctx[:, None, :]
        for m in methods:
            for cfg, groups in m.configs:
                t0 = time.time()
                kw = {}
                if groups:
                    kw["known_covariates"] = np.concatenate([batch.cov[g] for g in groups], axis=1)
                q = _forecast_batch(m.forecaster, ctx3, sc.horizon, kw, failures[m.name], cfg)
                stats[m.name][cfg][batch.stream, batch.origin] = window_stats(q, batch.y, batch.ctx, sc)
                timing[m.name] += time.time() - t0
        done += len(batch.stream)
        if (done // batch_size) % 10 == 0:
            log(f"  [{dataset.name}] {done}/{total} windows, {time.time() - t_start:.0f} s")
    log(f"  [{dataset.name}] finished {total} windows in {time.time() - t_start:.0f} s; forecast seconds: "
        + ", ".join(f"{k}={v:.0f}" for k, v in timing.items()))
    return {"stats": stats, "forecast_seconds": timing, "wall_seconds": time.time() - t_start, "failed_windows": failures}


def result_block(
    dataset,
    run: dict,
    methods: Sequence[MethodSpec],
    *,
    reps: int,
    seed: int,
    cluster: Optional[np.ndarray] = None,
) -> dict:
    """JSON-ready summary of one dataset: per method, per config, per bucket metrics with CIs and deltas vs none."""
    sc: Scale = dataset.scale
    labels = sc.labels()
    block = max(1, math.ceil(sc.horizon / dataset.origin_stride))
    out = {}
    for m in methods:
        sm = summarize(run["stats"][m.name], none_key="none", block=block, cluster=cluster, reps=reps, seed=seed)
        cfgs = {}
        for cfg, groups in m.configs:
            pt, bt = sm["point"][cfg], sm["boot"][cfg]
            lo, hi = ci(bt)
            entry = {"groups": list(groups), "buckets": {}}
            for j, lab in enumerate(labels):
                e = {"n_windows": int(np.isfinite(run["stats"][m.name][cfg][:, :, j, 0]).sum())}
                for k, mn in enumerate(METRIC_NAMES):
                    e[mn] = float(pt[j, k])
                    e[f"{mn}_ci95"] = [float(lo[j, k]), float(hi[j, k])]
                if "none" in sm["point"] and cfg != "none":
                    d = bt[:, j, :] - sm["boot"]["none"][:, j, :]
                    dl, dh = ci(d)
                    dpt = pt[j] - sm["point"]["none"][j]
                    rel = bt[:, j, 0] / sm["boot"]["none"][:, j, 0] - 1.0
                    rl, rh = ci(rel)
                    for k, mn in enumerate(METRIC_NAMES):
                        e[f"delta_{mn}"] = float(dpt[k])
                        e[f"delta_{mn}_ci95"] = [float(dl[k]), float(dh[k])]
                    e["rel_delta_wql"] = float(pt[j, 0] / sm["point"]["none"][j, 0] - 1.0)
                    e["rel_delta_wql_ci95"] = [float(rl), float(rh)]
                    if "boot_cluster" in sm:
                        dc = sm["boot_cluster"][cfg][:, j, 0] - sm["boot_cluster"]["none"][:, j, 0]
                        cl, ch = ci(dc)
                        e["delta_wql_cluster_ci95"] = [float(cl), float(ch)]
                entry["buckets"][lab] = e
            cfgs[cfg] = entry
        out[m.name] = {"configs": cfgs, "forecast_seconds": run["forecast_seconds"][m.name],
                       "failed_windows_by_config": run["failed_windows"][m.name]}
    return out
