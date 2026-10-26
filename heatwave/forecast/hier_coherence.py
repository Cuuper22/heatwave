"""Hierarchical forecast coherence adjustment (MinT, OLS, bottom-up).

Notation (Wickramasuriya, Athanasopoulos & Hyndman 2019, JASA 114(526):804-819,
doi:10.1080/01621459.2018.1448825): n series in the hierarchy, m bottom series,
summing matrix S (n x m) with y = S b. Adjusted (coherent) forecasts are y_tilde = S G y_hat with

* bottom-up : G = [0_(m x (n-m)) | I_m]
* OLS       : G = (S' S)^-1 S'
* MinT      : G = (S' W^-1 S)^-1 S' W^-1, W = covariance of the base forecast errors.

``shrink_covariance`` is the Schaefer-Strimmer (2005, Stat. Appl. Genet. Mol. Biol. 4(1):32,
doi:10.2202/1544-6115.1175) estimator used for MinT-shrink: the sample correlation matrix is
shrunk towards the identity with intensity lambda = sum_{i!=j} Var(r_ij) / sum_{i!=j} r_ij^2,
clipped to [0, 1]; variances are kept as estimated.

Approximation used for probabilistic forecasts (stated, not exact): the 21 quantile
levels (and the median) are adjusted one level at a time with the same G. A set of
level-q quantiles that adds up is not the same thing as the q-quantile of the sum, so the
adjusted quantiles are coherent as number vectors but are not the quantiles of a coherent
joint distribution. Adjusting each level separately can also make quantile curves cross
at a node; ``QuantileForecast`` sorts them, and ``coherence_error`` is reported both before
and after that sorting by the callers.

Array convention: the node axis is axis 1 of ``(N, n, H, Q)`` arrays (N windows, n nodes,
H horizon steps, Q quantile levels); median arrays are ``(N, n, H)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- hierarchy
@dataclass(frozen=True)
class Hierarchy:
    """A tree of aggregates over bottom series.

    ``names`` lists all n nodes with the aggregates first (top level first) and the m
    bottom series last. ``levels[i]`` is the level label of node i. ``S`` is (n, m).
    ``parent[i]`` is the index of the node's parent (-1 for roots).
    """

    names: tuple[str, ...]
    levels: tuple[str, ...]
    S: np.ndarray
    parent: tuple[int, ...]
    level_order: tuple[str, ...] = field(default=())

    @property
    def n(self) -> int:
        return self.S.shape[0]

    @property
    def m(self) -> int:
        return self.S.shape[1]

    @property
    def bottom_index(self) -> np.ndarray:
        return np.arange(self.n - self.m, self.n)

    def level_index(self, level: str) -> np.ndarray:
        return np.array([i for i, lv in enumerate(self.levels) if lv == level], dtype=int)

    def children(self, i: int) -> list[int]:
        return [j for j, p in enumerate(self.parent) if p == i]


def build_tree(
    root: str,
    mid: Mapping[str, Sequence[str]] | None,
    bottom_level: str = "gpu",
    root_level: str = "site",
    mid_level: str = "rack",
    bottoms: Sequence[str] | None = None,
) -> Hierarchy:
    """One tree: ``root`` -> (optional) mid nodes -> bottom series.

    * Two-level (e.g. node -> GPUs): ``mid=None`` and ``bottoms`` = the bottom names.
    * Three-level (site -> racks -> GPUs): ``mid={rack: [gpu, ...]}``.
    """
    if mid is None:
        if bottoms is None:
            raise ValueError("two-level tree needs bottoms")
        bot = list(bottoms)
        names = [root] + bot
        levels = [root_level] + [bottom_level] * len(bot)
        S = np.vstack([np.ones((1, len(bot))), np.eye(len(bot))])
        parent = [-1] + [0] * len(bot)
        return Hierarchy(tuple(names), tuple(levels), S, tuple(parent), (root_level, bottom_level))
    mids = list(mid.keys())
    bot = [b for k in mids for b in mid[k]]
    m = len(bot)
    names = [root] + mids + bot
    levels = [root_level] + [mid_level] * len(mids) + [bottom_level] * m
    S = np.zeros((len(names), m))
    S[0] = 1.0
    parent = [-1] + [0] * len(mids)
    col = 0
    for r, k in enumerate(mids):
        cnt = len(mid[k])
        S[1 + r, col:col + cnt] = 1.0
        parent += [1 + r] * cnt
        col += cnt
    S[1 + len(mids):] = np.eye(m)
    return Hierarchy(tuple(names), tuple(levels), S, tuple(parent), (root_level, mid_level, bottom_level))


# --------------------------------------------------------------------------- covariance
def shrink_covariance(resid: np.ndarray, variance_floor_rel: float = 1e-6) -> tuple[np.ndarray, float]:
    """Schaefer-Strimmer shrinkage covariance of ``resid`` (n_samples, n_series).

    Returns (W, lambda). Series with (near-)zero variance get the variance floor
    ``variance_floor_rel * median(positive variances)`` (1e-12 if all are zero) so that W stays
    positive definite; the floor is a numerical device, not a modelling choice.
    """
    x = np.asarray(resid, dtype=np.float64)
    x = x[np.all(np.isfinite(x), axis=1)]
    t, n = x.shape
    if t < 3:
        raise ValueError("need at least 3 residual rows")
    xc = x - x.mean(axis=0)
    var = (xc ** 2).sum(axis=0) / (t - 1)
    pos = var[var > 0]
    floor = variance_floor_rel * (np.median(pos) if pos.size else 1e-6)
    floor = max(floor, 1e-12)
    var = np.maximum(var, floor)
    sd = np.sqrt(var)
    z = xc / sd
    R = z.T @ z / (t - 1)
    # Var(r_ij) = t / (t-1)^3 * sum_k (w_kij - wbar_ij)^2 with w_kij = z_ki z_kj
    w_mean = R * (t - 1) / t
    zz = z ** 2
    s_w2 = (zz.T @ zz) - 2.0 * w_mean * (z.T @ z) + t * w_mean ** 2
    var_r = t / (t - 1) ** 3 * s_w2
    off = ~np.eye(n, dtype=bool)
    denom = float((R[off] ** 2).sum())
    lam = float(var_r[off].sum() / denom) if denom > 0 else 1.0
    lam = min(1.0, max(0.0, lam))
    Rs = (1.0 - lam) * R
    np.fill_diagonal(Rs, 1.0)
    W = Rs * np.outer(sd, sd)
    return W, lam


# --------------------------------------------------------------------------- G matrices
def g_bottom_up(S: np.ndarray) -> np.ndarray:
    n, m = S.shape
    G = np.zeros((m, n))
    G[:, n - m:] = np.eye(m)
    return G


def g_ols(S: np.ndarray) -> np.ndarray:
    return np.linalg.solve(S.T @ S, S.T)


def g_mint(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    Winv_S = np.linalg.solve(W, S)  # W^-1 S
    A = S.T @ Winv_S  # S' W^-1 S
    return np.linalg.solve(A, Winv_S.T)  # (S' W^-1 S)^-1 S' W^-1 (W symmetric)


def projection(S: np.ndarray, G: np.ndarray) -> np.ndarray:
    """P = S G (n x n): maps base forecasts to coherent forecasts."""
    return S @ G


def adjust(values: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply P along the node axis (axis 1) of (N, n, ...) arrays."""
    return np.einsum("ij,wj...->wi...", P, values, optimize=True)


# --------------------------------------------------------------------------- coherence
def coherence_sums(values: np.ndarray, h: Hierarchy, vs: str = "bottom") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per aggregate level: (numerator, denominator) of the relative L1 incoherence, separately for every entry of the
    last axis (the quantile axis; pass a trailing axis of size 1 for medians). Sums run over windows, steps and the
    nodes of that level, so results from chunks of windows can be added before dividing.

    ``vs='bottom'`` compares an aggregate with the sum of the bottom series below it (S_A b); ``vs='children'`` with
    the sum of its immediate children.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    b = values[:, h.bottom_index]
    axes = tuple(range(values.ndim - 1))
    for lv in h.level_order[:-1]:
        num = 0.0
        den = 0.0
        for i in h.level_index(lv):
            if vs == "bottom":
                agg = np.einsum("j,wj...->w...", h.S[i], b)
            else:
                agg = values[:, h.children(i)].sum(axis=1)
            num = num + np.abs(values[:, i] - agg).sum(axis=tuple(range(values[:, i].ndim - 1)))
            den = den + np.abs(values[:, i]).sum(axis=tuple(range(values[:, i].ndim - 1)))
        out[lv] = (np.atleast_1d(num), np.atleast_1d(den))
    return out


def coherence_error(values: np.ndarray, h: Hierarchy, vs: str = "bottom") -> dict[str, float]:
    """Relative L1 incoherence per aggregate level, pooled over all trailing axes:
    sum |y_A - sum of descendants| / sum |y_A| (0 for coherent forecasts)."""
    out = {}
    for lv, (num, den) in coherence_sums(values, h, vs).items():
        out[lv] = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    return out


__all__ = ["Hierarchy", "build_tree", "shrink_covariance", "g_bottom_up", "g_ols", "g_mint", "projection", "adjust",
           "coherence_sums", "coherence_error"]
