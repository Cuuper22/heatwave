"""Shared helpers for the forecasting studies: forecaster set, cached forecasting, scoring and bootstrap.

Scores use ``heatwave.forecast.metrics`` (pinball / WQL, MAE and MASE of the median, central-interval coverage)
on blocks of windows. The dataset-level numbers are aggregates over windows (origins):

* WQL   = 2 * mean pinball (21 levels) / mean |actual|, both averaged over the windows in the aggregate
          (ratio of means, so near-zero actuals in single windows cannot blow it up)
* MASE  = mean over windows of MAE(median) / in-context seasonal-naive MAE (scale from the 256/96-step context)
* MAE   = mean over windows of MAE(median) in the unit of the series
* cov80 / cov98 = mean over windows of the share of actuals inside the 10-90% / 1-99% quantile interval

Confidence intervals: 95% percentile bootstrap, 1000 resamples of the windows (origins) with replacement (``ci_origin``,
the interval the task asks for), and, because windows from one series or one run are correlated, also a bootstrap that
resamples whole groups (session / run / job / seed; ``ci_group``).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heatwave.forecast import ARIMAForecaster, ETSForecaster, QUANTILE_LEVELS, QuantileForecast, SeasonalNaiveForecaster  # noqa: E402
from heatwave.forecast import metrics as M  # noqa: E402
from heatwave.forecast.classical import _PerSeries, _from_intervals, _interval_levels  # noqa: E402
from heatwave.forecast.datasets import CACHE_DIR, SEASON  # noqa: E402

N_BOOT = 1000
BOOT_SEED = 20261026
ALL_METHODS = ("timestral-5M", "timestral-28M", "timestral-209M", "seasonal-naive", "ets", "ets-flat", "arima")
FC_CACHE = CACHE_DIR / "fc"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hardware_note() -> str:
    return f"{platform.platform()}; cpu_count={os.cpu_count()}; python {platform.python_version()}"


# --------------------------------------------------------------------------- forecasters
class ETSFlatForecaster(_PerSeries):
    """AutoETS restricted to additive errors and no trend (statsforecast model 'ANN', 'ANA' with a season).

    Added because the shared ``ETSForecaster`` (unrestricted AutoETS) selected ETS(M,A,N) on many windows of the
    seconds-scale power series, whose multiplicative-error trend intervals reach 1e12 and above. Both are reported.
    """

    def __init__(self, season_length: int = 1, max_context: int | None = 2048):
        self.season_length = season_length
        self.max_context = max_context
        self.name = f"ets-flat-{season_length}"

    def _one(self, y: np.ndarray, horizon: int) -> np.ndarray:
        from statsforecast.models import AutoETS

        try:
            r = AutoETS(season_length=self.season_length, model="ANN" if self.season_length == 1 else "ANA").forecast(
                y=y, h=horizon, level=_interval_levels())
            q = _from_intervals(r, horizon)
            if np.all(np.isfinite(q)):
                return q
        except Exception:
            pass
        return SeasonalNaiveForecaster(max(1, self.season_length))._one(y, horizon)


def make_classical(kind: str, scale: str):
    """Classical baselines with orders fixed a priori per scale (no tuning)."""
    s = SEASON[scale]
    if kind == "seasonal-naive":
        return SeasonalNaiveForecaster(s)
    if kind == "ets":
        return ETSForecaster(season_length=s)
    if kind == "ets-flat":
        return ETSFlatForecaster(season_length=s)
    if kind == "arima":
        if s > 1:
            return ARIMAForecaster(order=(2, 1, 1), season_length=s, seasonal_order=(1, 0, 0))
        return ARIMAForecaster(order=(2, 1, 1))
    raise ValueError(kind)


def make_timestral(label: str, threads: int, weights_dir: Optional[str] = None):
    from heatwave.forecast.timestral_fc import TimestralForecaster

    size = label.split("-")[1]
    return TimestralForecaster(size, weights_dir=weights_dir, threads=threads)


def _classical_chunk(args):
    kind, scale, ctx, horizon = args
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    f = make_classical(kind, scale)
    return f.forecast(ctx[:, None, :], horizon).values.astype(np.float32)


def ctx_hash(arr: np.ndarray, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(extra.encode())
    return h.hexdigest()[:16]


def forecast_series(
    method: str, scale: str, context: np.ndarray, horizon: int, *, threads: int = 2, workers: int = 1, batch: int = 64,
    weights_dir: Optional[str] = None, cache_key: Optional[str] = None, timestral=None,
) -> tuple[np.ndarray, float, bool]:
    """Forecast every row of ``context`` (N, T) independently with ``method``.

    Returns (values (N, 1, H, 21) float32, seconds per series, from_cache). A cache file keyed by
    ``cache_key`` (plus a hash of the context) makes interrupted runs resumable.
    """
    N = context.shape[0]
    path = None
    h = ctx_hash(context, f"{method}|{scale}|{horizon}|{weights_dir}")
    if cache_key:
        FC_CACHE.mkdir(parents=True, exist_ok=True)
        path = FC_CACHE / f"{cache_key}__{method}.npz"
        if path.exists():
            z = np.load(path)
            if str(z["hash"]) == h:
                return z["values"], float(z["sec_per_series"]), True
    t0 = time.perf_counter()
    if method.startswith("timestral"):
        f = timestral if timestral is not None else make_timestral(method, threads, weights_dir)
        parts = []
        for i in range(0, N, batch):
            parts.append(f.forecast(context[i:i + batch, None, :], horizon).values.astype(np.float32))
        vals = np.concatenate(parts, axis=0)
    else:
        if workers > 1 and N >= 2 * workers:
            chunks = np.array_split(np.arange(N), workers * 4)
            ctxm = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctxm) as ex:
                parts = list(ex.map(_classical_chunk, [(method, scale, context[c], horizon) for c in chunks if len(c)]))
            vals = np.concatenate(parts, axis=0)
        else:
            vals = _classical_chunk((method, scale, context, horizon))
    sec = (time.perf_counter() - t0) / max(N, 1)
    if path is not None:
        np.savez(path, values=vals, hash=np.asarray(h), sec_per_series=np.asarray(sec))
    return vals, sec, False


# --------------------------------------------------------------------------- scoring
COMP_KEYS = ("pin", "absy", "mae", "mase", "cov80", "cov98")


def score_block(vals: np.ndarray, fut: np.ndarray, ctx: np.ndarray, season: int, lo: int, hi: int) -> dict[str, float]:
    """Metrics of one block: vals (B, 1, H, 21), fut (B, H), ctx (B, T); horizon steps lo..hi (1-based, inclusive)."""
    fc = QuantileForecast(values=np.asarray(vals[:, :, lo - 1:hi, :], dtype=np.float64))
    y = np.asarray(fut[:, None, lo - 1:hi], dtype=np.float64)
    c = np.asarray(ctx[:, None, :], dtype=np.float64)
    with np.errstate(all="ignore"):
        mase = M.mase(fc, y, c, season)
    return {
        "pin": M.pinball(fc, y), "absy": float(np.nanmean(np.abs(y))), "mae": M.mae_median(fc, y), "mase": mase,
        "cov80": M.coverage(fc, y, 0.8), "cov98": M.coverage(fc, y, 0.98),
    }


def score_windows(vals: np.ndarray, fut: np.ndarray, ctx: np.ndarray, season: int, buckets: Sequence[tuple[str, int, int]]) -> dict[str, dict[str, np.ndarray]]:
    """Per-window components for each horizon bucket (plus 'all')."""
    H = fut.shape[1]
    bl = list(buckets) + [("all", 1, H)]
    out = {b[0]: {k: np.empty(vals.shape[0]) for k in COMP_KEYS} for b in bl}
    for i in range(vals.shape[0]):
        for label, lo, hi in bl:
            r = score_block(vals[i:i + 1], fut[i:i + 1], ctx[i:i + 1], season, lo, min(hi, H))
            for k in COMP_KEYS:
                out[label][k][i] = r[k]
    return out


# --------------------------------------------------------------------------- bootstrap
def _weights(n: int, group: Optional[np.ndarray], n_boot: int, seed: int, by_group: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if not by_group or group is None:
        return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
    _, gi = np.unique(np.asarray(group, dtype=str), return_inverse=True)
    G = gi.max() + 1
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(np.float64)
    return counts[:, gi]


def _stats(W: np.ndarray, comp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pin, absy = np.asarray(comp["pin"]), np.asarray(comp["absy"])
    mase = np.asarray(comp["mase"])
    ok = np.isfinite(mase)
    with np.errstate(all="ignore"):
        return {
            "wql": 2.0 * (W @ pin) / (W @ absy),
            "mase": (W @ np.where(ok, mase, 0.0)) / (W @ ok.astype(float)),
            "mae": (W @ np.asarray(comp["mae"])) / W.sum(axis=1),
            "cov80": (W @ np.asarray(comp["cov80"])) / W.sum(axis=1),
            "cov98": (W @ np.asarray(comp["cov98"])) / W.sum(axis=1),
        }


def bootstrap_summary(comp: dict[str, np.ndarray], group: Optional[np.ndarray] = None, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    n = len(comp["pin"])
    est = _stats(np.ones((1, n)), comp)
    Wo = _weights(n, group, n_boot, seed, False)
    so = _stats(Wo, comp)
    n_groups = int(len(np.unique(np.asarray(group, dtype=str)))) if group is not None else 0
    sg = _stats(_weights(n, group, n_boot, seed + 1, True), comp) if n_groups > 1 else None
    out = {}
    for k, v in est.items():
        d = {"est": float(v[0])}
        d["lo"], d["hi"] = (float(x) for x in np.nanpercentile(so[k], [2.5, 97.5]))
        if sg is not None:
            d["lo_group"], d["hi_group"] = (float(x) for x in np.nanpercentile(sg[k], [2.5, 97.5]))
        out[k] = d
    out["n_windows"] = int(n)
    out["n_groups"] = n_groups
    out["n_mase_valid"] = int(np.isfinite(np.asarray(comp["mase"])).sum())
    return out


def bootstrap_delta(comp_a: dict[str, np.ndarray], comp_b: dict[str, np.ndarray], group: Optional[np.ndarray] = None,
                    n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Paired difference a - b of every statistic (same resampled windows for both)."""
    n = len(comp_a["pin"])
    Wo = _weights(n, group, n_boot, seed, False)
    ea, eb = _stats(np.ones((1, n)), comp_a), _stats(np.ones((1, n)), comp_b)
    sa, sb = _stats(Wo, comp_a), _stats(Wo, comp_b)
    out = {}
    for k in ea:
        d = {"est": float(ea[k][0] - eb[k][0])}
        d["lo"], d["hi"] = (float(x) for x in np.nanpercentile(sa[k] - sb[k], [2.5, 97.5]))
        out[k] = d
    return out
