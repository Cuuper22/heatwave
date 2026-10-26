"""Shared forecast types for Heatwave.

Every forecaster in this package (Timestral, the classical baselines and the
perfect-foresight oracle) returns the same object: 21 quantiles per series,
variate and horizon step. Decisions downstream only ever see this object, so a
forecaster can be swapped without touching the decision code.

Shapes follow Timestral's convention:
  context            (B, V, T)      B series, V variates, T past steps; NaN = missing
  known_covariates   (B, C, T + H)  known over the past and the horizon
  past_covariates    (B, C, T)      known over the past only
  QuantileForecast.values  (B, V, H, Q) with Q = 21
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

# Timestral's 21 output quantile levels. The classical baselines and the
# oracle are mapped onto the same grid so every method is scored identically.
QUANTILE_LEVELS: tuple[float, ...] = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)
MEDIAN_INDEX = QUANTILE_LEVELS.index(0.5)


def as_bvt(context: np.ndarray) -> np.ndarray:
    """Coerce (T,), (B, T) or (B, V, T) to float64 (B, V, T)."""
    x = np.asarray(context, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, None, :]
    elif x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim != 3:
        raise ValueError(f"context must have 1-3 dims, got shape {x.shape}")
    return x


@dataclass(frozen=True)
class QuantileForecast:
    """21-quantile forecast, values shaped (B, V, H, Q)."""

    values: np.ndarray
    levels: tuple[float, ...] = QUANTILE_LEVELS
    method: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        if v.ndim != 4 or v.shape[-1] != len(self.levels):
            raise ValueError(
                f"values must be (B, V, H, {len(self.levels)}), got {v.shape}"
            )
        # Enforce non-crossing quantiles (sorting is the standard fix and is a
        # no-op for Timestral, whose head is already monotone).
        object.__setattr__(self, "values", np.sort(v, axis=-1))

    @property
    def median(self) -> np.ndarray:
        return self.values[..., MEDIAN_INDEX]

    @property
    def horizon(self) -> int:
        return self.values.shape[2]

    def quantile(self, level: float) -> np.ndarray:
        """Quantile at any level in [0.01, 0.99], linear between grid levels."""
        lv = np.asarray(self.levels)
        if not lv[0] <= level <= lv[-1]:
            raise ValueError(f"level {level} outside [{lv[0]}, {lv[-1]}]")
        j = int(np.searchsorted(lv, level))
        if j < len(lv) and np.isclose(lv[j], level):
            return self.values[..., j]
        lo, hi = j - 1, j
        w = (level - lv[lo]) / (lv[hi] - lv[lo])
        return (1 - w) * self.values[..., lo] + w * self.values[..., hi]

    def interval(self, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1 - coverage) / 2
        return self.quantile(a), self.quantile(1 - a)


@runtime_checkable
class Forecaster(Protocol):
    """Anything that turns a past window into a 21-quantile forecast."""

    name: str

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        *,
        known_covariates: Optional[np.ndarray] = None,
        past_covariates: Optional[np.ndarray] = None,
    ) -> QuantileForecast:
        ...
