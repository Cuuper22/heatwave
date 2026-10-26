"""Classical forecast baselines on the shared 21-quantile grid.

* SeasonalNaiveForecaster: repeat the last season; quantiles from the empirical
  distribution of in-sample seasonal-naive errors, widened by sqrt(k) for the
  k-th season ahead.
* ETSForecaster: statsforecast AutoETS with its prediction intervals.
* ARIMAForecaster: statsforecast ARIMA with a fixed small order (AutoARIMA
  took about 100 s per series on this VM, too slow for rolling evaluation).

Classical baselines ignore covariates (they are univariate by design); each
variate of each series is forecast independently.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import MEDIAN_INDEX, QUANTILE_LEVELS, QuantileForecast, as_bvt

_LEVELS = np.asarray(QUANTILE_LEVELS)


def _ffill_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    if np.all(np.isnan(y)):
        return np.zeros_like(y)
    first = np.flatnonzero(~np.isnan(y))[0]
    y[:first] = y[first]
    for i in range(1, len(y)):
        if np.isnan(y[i]):
            y[i] = y[i - 1]
    return y


def _interval_levels() -> list[int]:
    """Central interval levels (percent) whose bounds land on QUANTILE_LEVELS."""
    out = []
    for q in QUANTILE_LEVELS:
        if q < 0.5:
            out.append(int(round((1 - 2 * q) * 100)))
    return sorted(set(out))


def _from_intervals(result: dict, horizon: int) -> np.ndarray:
    """Map statsforecast {'mean','lo-L','hi-L'} onto the 21-quantile grid."""
    q = np.empty((horizon, len(QUANTILE_LEVELS)))
    for j, level in enumerate(QUANTILE_LEVELS):
        if np.isclose(level, 0.5):
            q[:, j] = result["mean"]
        elif level < 0.5:
            q[:, j] = result[f"lo-{int(round((1 - 2 * level) * 100))}"]
        else:
            q[:, j] = result[f"hi-{int(round((2 * level - 1) * 100))}"]
    return q


class _PerSeries:
    name = "base"
    max_context: Optional[int] = None

    def _one(self, y: np.ndarray, horizon: int) -> np.ndarray:  # (H, Q)
        raise NotImplementedError

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        *,
        known_covariates: Optional[np.ndarray] = None,
        past_covariates: Optional[np.ndarray] = None,
    ) -> QuantileForecast:
        ctx = as_bvt(context)
        if self.max_context is not None:
            ctx = ctx[..., -self.max_context:]
        B, V, _ = ctx.shape
        out = np.empty((B, V, horizon, len(QUANTILE_LEVELS)))
        for b in range(B):
            for v in range(V):
                out[b, v] = self._one(_ffill_nan(ctx[b, v]), horizon)
        return QuantileForecast(values=out, method=self.name)


class SeasonalNaiveForecaster(_PerSeries):
    def __init__(self, season_length: int, max_context: Optional[int] = None):
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length
        self.max_context = max_context
        self.name = f"seasonal-naive-{season_length}"

    def _one(self, y: np.ndarray, horizon: int) -> np.ndarray:
        s = min(self.season_length, len(y) - 1) if len(y) > 1 else 1
        last = y[-s:]
        steps = np.arange(horizon)
        point = last[steps % s]
        if len(y) > s + 1:
            err = y[s:] - y[:-s]
        else:
            err = np.zeros(1)
        base_q = np.quantile(err, _LEVELS)  # (Q,)
        k = (steps // s + 1).astype(np.float64)
        q = point[:, None] + np.sqrt(k)[:, None] * base_q[None, :]
        # Centre the median on the point forecast.
        q += (point - q[:, MEDIAN_INDEX])[:, None]
        return q


class ETSForecaster(_PerSeries):
    def __init__(self, season_length: int = 1, max_context: Optional[int] = 2048):
        self.season_length = season_length
        self.max_context = max_context
        self.name = f"ets-{season_length}"

    def _one(self, y: np.ndarray, horizon: int) -> np.ndarray:
        from statsforecast.models import AutoETS

        try:
            r = AutoETS(season_length=self.season_length).forecast(
                y=y, h=horizon, level=_interval_levels()
            )
            q = _from_intervals(r, horizon)
            if np.all(np.isfinite(q)):
                return q
        except Exception:
            pass
        return SeasonalNaiveForecaster(max(1, self.season_length))._one(y, horizon)


class ARIMAForecaster(_PerSeries):
    def __init__(
        self,
        order: tuple[int, int, int] = (2, 1, 1),
        season_length: int = 1,
        seasonal_order: tuple[int, int, int] = (0, 0, 0),
        max_context: Optional[int] = 1024,
    ):
        self.order = order
        self.season_length = season_length
        self.seasonal_order = seasonal_order
        self.max_context = max_context
        self.name = f"arima-{order[0]}{order[1]}{order[2]}"

    def _one(self, y: np.ndarray, horizon: int) -> np.ndarray:
        from statsforecast.models import ARIMA

        try:
            r = ARIMA(
                order=self.order,
                season_length=self.season_length,
                seasonal_order=self.seasonal_order,
            ).forecast(y=y, h=horizon, level=_interval_levels())
            q = _from_intervals(r, horizon)
            if np.all(np.isfinite(q)):
                return q
        except Exception:
            pass
        return SeasonalNaiveForecaster(max(1, self.season_length))._one(y, horizon)


class PerfectForesight:
    """Upper-bound oracle: every quantile equals the realised future.

    Call ``set_future(future)`` with the (B, V, H) realised values before each
    ``forecast``. Used only as an upper bound for decisions, never scored as a
    forecaster.
    """

    name = "perfect-foresight"

    def __init__(self) -> None:
        self._future: Optional[np.ndarray] = None

    def set_future(self, future: np.ndarray) -> None:
        self._future = as_bvt(future)

    def forecast(self, context, horizon, *, known_covariates=None, past_covariates=None):
        if self._future is None or self._future.shape[-1] < horizon:
            raise RuntimeError("PerfectForesight needs set_future() with >= horizon steps")
        f = self._future[..., :horizon]
        return QuantileForecast(
            values=np.repeat(f[..., None], len(QUANTILE_LEVELS), axis=-1),
            method=self.name,
        )
