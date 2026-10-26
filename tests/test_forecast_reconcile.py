"""Tests for hierarchy coherence adjustment (bottom-up, OLS, MinT shrinkage)."""

import numpy as np
import pytest

from heatwave.forecast import hier_coherence as C


def tree():
    return C.build_tree("site", {"r1": ["a", "b"], "r2": ["c", "d", "e"]}, bottom_level="gpu")


def test_summing_matrix_and_levels():
    h = tree()
    assert h.S.shape == (8, 5)
    assert list(h.S[0]) == [1, 1, 1, 1, 1]
    assert list(h.S[1]) == [1, 1, 0, 0, 0]
    assert h.levels[0] == "site" and h.levels[1] == "rack" and h.levels[-1] == "gpu"
    assert h.parent[3] == 1 and h.parent[5] == 2


def test_two_level_tree():
    h = C.build_tree("node", None, root_level="node", bottoms=["g0", "g1", "g2"])
    assert h.S.shape == (4, 3) and h.level_order == ("node", "gpu")


@pytest.mark.parametrize("which", ["bu", "ols", "mint"])
def test_adjusted_forecasts_are_coherent_and_projection_is_idempotent(which):
    h = tree()
    rng = np.random.default_rng(1)
    base = rng.normal(10, 2, (30, h.n, 7, 3))
    W, _ = C.shrink_covariance(rng.normal(size=(200, h.n)))
    G = {"bu": C.g_bottom_up(h.S), "ols": C.g_ols(h.S), "mint": C.g_mint(h.S, W)}[which]
    P = C.projection(h.S, G)
    out = C.adjust(base, P)
    assert max(C.coherence_error(out, h).values()) < 1e-12
    assert np.allclose(P @ P, P)
    assert max(C.coherence_error(base, h).values()) > 0.01


def test_coherent_input_is_unchanged_by_every_method():
    h = tree()
    rng = np.random.default_rng(2)
    b = rng.normal(5, 1, (10, h.m, 4, 1))
    y = np.einsum("ij,wjhq->wihq", h.S, b)
    W, _ = C.shrink_covariance(rng.normal(size=(100, h.n)))
    for G in (C.g_bottom_up(h.S), C.g_ols(h.S), C.g_mint(h.S, W)):
        assert np.allclose(C.adjust(y, C.projection(h.S, G)), y)


def test_bottom_up_keeps_bottom_and_mint_with_identity_w_is_ols():
    h = tree()
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, (4, h.n, 3, 1))
    out = C.adjust(base, C.projection(h.S, C.g_bottom_up(h.S)))
    assert np.allclose(out[:, h.bottom_index], base[:, h.bottom_index])
    assert np.allclose(C.g_mint(h.S, np.eye(h.n)), C.g_ols(h.S))


def test_shrinkage_covariance_properties():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, 6)) * np.array([1, 2, 3, 4, 5, 6.0])
    W, lam = C.shrink_covariance(x)
    assert 0.0 <= lam <= 1.0
    assert np.allclose(W, W.T) and np.linalg.eigvalsh(W).min() > 0
    assert np.allclose(np.diag(W), x.var(axis=0, ddof=1))
    # strongly correlated, plenty of samples: little shrinkage
    z = rng.normal(size=(5000, 1))
    corr = np.hstack([z + 0.1 * rng.normal(size=(5000, 1)) for _ in range(4)])
    _, lam2 = C.shrink_covariance(corr)
    assert lam2 < 0.05


def test_zero_variance_series_gets_floor_not_singular():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(50, 4))
    x[:, 2] = 0.0
    W, _ = C.shrink_covariance(x)
    assert np.all(np.isfinite(W)) and np.linalg.eigvalsh(W).min() > 0


def test_coherence_error_children_vs_bottom():
    h = tree()
    y = np.zeros((1, h.n, 1, 1))
    y[0, h.bottom_index, 0, 0] = [1, 2, 3, 4, 5]
    y[0, 1] = 3
    y[0, 2] = 12
    y[0, 0] = 15
    assert C.coherence_error(y, h)["rack"] == pytest.approx(0.0)
    assert C.coherence_error(y, h)["site"] == pytest.approx(0.0)
    y[0, 0] = 30
    assert C.coherence_error(y, h)["site"] == pytest.approx(0.5)
    assert C.coherence_error(y, h, "children")["site"] == pytest.approx(0.5)
