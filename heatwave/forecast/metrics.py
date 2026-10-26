"""Forecast scoring shared by every experiment.

* pinball / weighted quantile loss (WQL): mean pinball loss over the 21 levels,
  the standard CRPS approximation used by GIFT-Eval and fev-bench.
* MASE of the median, scaled by the in-sample seasonal-naive error.
* empirical coverage of central intervals.
"""

from __future__ import annotations

import numpy as np

from .base import QuantileForecast


def pinball(fc: QuantileForecast, actual: np.ndarray) -> float:
    """Mean pinball loss over all quantiles, series, variates and steps."""
    y = np.asarray(actual, dtype=np.float64)[..., None]  # (B,V,H,1)
    q = fc.values
    lv = np.asarray(fc.levels)
    diff = y - q
    loss = np.maximum(lv * diff, (lv - 1) * diff)
    return float(np.nanmean(loss))


def wql(fc: QuantileForecast, actual: np.ndarray) -> float:
    """Pinball loss scaled by mean |actual| (scale-free, as in GIFT-Eval)."""
    y = np.asarray(actual, dtype=np.float64)
    denom = np.nanmean(np.abs(y))
    return 2.0 * pinball(fc, y) / denom if denom > 0 else float("nan")


def mae_median(fc: QuantileForecast, actual: np.ndarray) -> float:
    return float(np.nanmean(np.abs(fc.median - np.asarray(actual, dtype=np.float64))))


def mase(fc: QuantileForecast, actual: np.ndarray, context: np.ndarray, season: int = 1) -> float:
    ctx = np.asarray(context, dtype=np.float64)
    scale = np.nanmean(np.abs(ctx[..., season:] - ctx[..., :-season]))
    return mae_median(fc, actual) / scale if scale > 0 else float("nan")


def coverage(fc: QuantileForecast, actual: np.ndarray, level: float = 0.8) -> float:
    lo, hi = fc.interval(level)
    y = np.asarray(actual, dtype=np.float64)
    ok = (y >= lo) & (y <= hi)
    return float(np.nanmean(ok))
