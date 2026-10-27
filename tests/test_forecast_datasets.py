import pytest
"""Tests for window cutting and the fixed splits of the forecasting studies."""

import numpy as np
import pandas as pd

from heatwave.forecast import datasets as D

from heatwave.data.schema import PROCESSED
# MIT Supercloud series are not redistributed (CC BY-NC-ND 4.0); rebuild them with scripts/fetch_mit_supercloud.py
needs_mit = pytest.mark.skipif(not (PROCESSED / "public_gpu_mit_jobs.parquet").exists(),
                               reason="MIT Supercloud series not present; run scripts/fetch_mit_supercloud.py")


def test_origin_positions_bounds_and_thinning():
    pos, n = D.origin_positions(1000, 256, 300, 50)
    assert pos[0] == 256 and pos[-1] <= 700 and n == len(pos)
    pos2, n2 = D.origin_positions(1000, 256, 300, 50, max_n=4)
    assert n2 == n and len(pos2) == 4 and pos2[0] == pos[0] and pos2[-1] == pos[-1]
    assert D.origin_positions(100, 256, 300, 10)[1] == 0
    p3, _ = D.origin_positions(1000, 256, 300, 50, first=500)
    assert p3[0] >= 500


def test_make_windows_shapes_and_leakage_free():
    y = np.arange(2000, dtype=float)
    s = D.Series(id="x", values=y, step_s=1.0, t0=pd.Timestamp("2026-01-01", tz="UTC"), group="g",
                 cov=np.vstack([y, -y]), cov_names=("a", "b"))
    w = D.make_windows("t", "seconds", "public", [s], stride=100)
    assert w.context.shape[1] == 256 and w.future.shape[1] == 300
    assert np.all(w.future[:, 0] == w.origin_idx) and np.all(w.context[:, -1] == w.origin_idx - 1)
    assert w.known_cov.shape == (w.n, 2, 556)
    assert w.origin_time[0].startswith("2026-01-01T00:04:16")


def test_windows_with_missing_future_are_dropped():
    y = np.arange(1500, dtype=float)
    y[700] = np.nan
    w = D.make_windows("t", "seconds", "public", [D.Series(id="x", values=y, step_s=1.0)], stride=50)
    assert w.meta["n_dropped_nonfinite"] > 0
    assert np.all(np.isfinite(w.future))


def test_subsample_index_is_deterministic_and_spans_range():
    i = D.subsample_index(100, 5)
    assert list(i) == list(D.subsample_index(100, 5)) and i[0] == 0 and i[-1] == 99 and len(i) == 5
    assert len(D.subsample_index(3, 5)) == 3


@needs_mit
def test_splits_are_disjoint_and_follow_the_rules():
    te, tr = D.node_sessions("test"), D.node_sessions("train")
    assert not set(te) & set(tr) and len(te) + len(tr) == 32
    assert not set(D.nlr_run_names("test")) & set(D.nlr_run_names("train"))
    assert all(j % 4 == 0 for j in D.mit_job_ids("test")) and all(j % 4 != 0 for j in D.mit_job_ids("train"))
    assert D.SIM_TEST_SEEDS == tuple(range(100, 110)) and max(D.SIM_VAL_SEEDS) < 60


def test_grid_series_test_origins_start_in_may_2026():
    s = D.caiso_price_series(False)[0]
    w = D.make_windows("c", "hours", "public", [s], stride=24)
    assert pd.Timestamp(w.origin_time[0]) >= D.TEST_FROM_UTC


def test_public_hierarchy_node_is_exact_sum():
    hw = D.hier_public_node("test", stride=300)
    assert hw.context.shape[1] == 9
    assert np.allclose(hw.context[:, 0], hw.context[:, 1:].sum(axis=1))
    assert np.allclose(hw.future[:, 0], hw.future[:, 1:].sum(axis=1))
