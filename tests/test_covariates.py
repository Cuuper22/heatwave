"""Leakage tests for every covariate builder, plus the ablation engine's agreement with heatwave.forecast.metrics.

Each leakage test changes something that is NOT available at the forecast origin (jobs submitted later, source values
that are published later, realised checkpoints, the actual weather) and asserts that the covariate window does not
change; where it can, it also shows that the change would matter if it were let through.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ablation"))

from heatwave.forecast import QUANTILE_LEVELS, QuantileForecast
from heatwave.forecast import covariates as cv
from heatwave.forecast.metrics import coverage, mase, pinball, wql


# ----------------------------------------------------------------------------------------------- simulator builders
@pytest.fixture(scope="module")
def sim():
    hs = pytest.importorskip("gpu_stack.research.heatwave_sim")
    path = Path(cv.SCENARIO_PATH)
    if not path.exists():
        pytest.skip("simulator scenario not available")
    return hs, json.loads(path.read_text())


@pytest.fixture(scope="module")
def builder(sim):
    return cv.SimCovariateBuilder(sim[1], 100, "worlds")


def test_world_scenario_variants(sim):
    sc = sim[1]
    lit = cv.world_scenario(sc, 105, "literal")
    assert lit["jobs"]["generate"]["seed"] == sc["jobs"]["generate"]["seed"]
    wor = cv.world_scenario(sc, 105, "worlds")
    assert wor["jobs"]["generate"]["seed"] == 105 and wor["environment"]["seed"] == 105
    assert sc["jobs"]["generate"]["seed"] != 105  # the input is not modified


def test_sim_plan_ignores_jobs_submitted_after_origin(sim, builder):
    hs, sc = sim
    late_job = next(
        j for j in builder._jobs
        if j.submit_s > 72 * 3600 and j.start_s <= j.submit_s + 6 * 3600 and j.submit_s < 140 * 3600
    )
    origin_h = int(late_job.submit_s // 3600)
    o_s = origin_h * 3600.0
    assert o_s < late_job.submit_s <= late_job.start_s <= o_s + 24 * 3600
    base = builder.window("site", 3600.0, origin_h, 72, 24)
    late = [j for j in builder._jobs if j.submit_s > o_s]
    assert late, "scenario must have jobs submitted after the origin for this test to mean anything"
    # change every late job: the covariate at the origin must not move
    b2 = cv.SimCovariateBuilder(sc, 100, "worlds")
    b2._jobs = [
        (j if j.submit_s <= o_s else type(j)(**{**j.to_dict(), "duration_s": j.duration_s * 2.0, "n_gpus": j.n_gpus}))
        for j in b2._jobs
    ]
    b2._submit = np.array([j.submit_s for j in b2._jobs])
    changed = b2.window("site", 3600.0, origin_h, 72, 24)
    for g in ("J", "P", "W", "C"):
        np.testing.assert_array_equal(base[g], changed[g])
    # the plan that knows every job differs in the horizon (the test has power) and agrees on the past
    full = builder.window("site", 3600.0, origin_h, 72, 24, full_knowledge=True)
    assert not np.array_equal(full["J"][..., 72:], base["J"][..., 72:])
    np.testing.assert_allclose(full["J"][..., :72], base["J"][..., :72])
    np.testing.assert_allclose(full["P"][..., :72], base["P"][..., :72])


def test_sim_full_knowledge_equals_simulator_covariates(sim, builder):
    hs, _ = sim
    plans = hs.plan_schedule(builder.cfg)
    ref = hs.planned_covariates(builder.cfg, plans, 3600.0, "site")
    w = builder.window("site", 3600.0, 100, 50, 24, full_knowledge=True)
    for g, keys in cv.SIM_GROUPS.items():
        for c, k in enumerate(keys):
            np.testing.assert_allclose(w[g][c].T, ref[k][50:50 + 74])


def test_sim_weather_is_the_forecast_not_the_actual(sim):
    hs, sc = sim
    b = cv.SimCovariateBuilder(sc, 101, "worlds")
    before = b.window("site", 3600.0, 100, 72, 24)["W"].copy()
    for env in b.cfg.environment.values():
        assert env.outdoor_forecast_c is not None
        env.outdoor_c[:] = env.outdoor_c + 10.0  # change the actual outdoor temperature
    after = b.window("site", 3600.0, 100, 72, 24)["W"]
    np.testing.assert_array_equal(before, after)
    # and it is not equal to the actual
    for u, env in enumerate(b.cfg.environment.values()):
        assert not np.allclose(after[0, u, :], env.outdoor_c[28:124])


def test_sim_covariates_do_not_depend_on_the_noise_seed(sim):
    _, sc = sim
    a = cv.SimCovariateBuilder(sc, 100, "literal").window("site", 3600.0, 110, 72, 24)
    b = cv.SimCovariateBuilder(sc, 101, "literal").window("site", 3600.0, 110, 72, 24)
    for g in a:
        np.testing.assert_array_equal(a[g], b[g])


def test_sim_horizon_limit(builder):
    with pytest.raises(cv.LeakageError):
        builder.window("site", 3600.0, 100, 72, 25)  # 25 h > known-future horizon 24 h
    builder.window("site", 3600.0, 100, 72, 24)


# -------------------------------------------------------------------------------------------------- issue-time builders
def _grid(start="2026-06-10 00:00", end="2026-06-14 00:00", freq="5min"):
    return pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")


def test_day_ahead_price_publication_rule():
    grid = _grid()
    dam = pd.Series(np.arange(24 * 4, dtype=float) + 100.0, index=pd.date_range("2026-06-10", periods=24 * 4, freq="1h", tz="UTC"))
    c = cv.day_ahead_price_covariate(dam, grid)
    T, H = 12, 288
    # 12:55 Pacific on 2026-06-11 (PDT, UTC-7) = 19:55 UTC: day-ahead for 06-12 is not published yet
    i = int(grid.get_loc(pd.Timestamp("2026-06-11 19:55", tz="UTC")))
    w = c.window(i, T, H)[0]
    local_day = grid[i - T:i + H].tz_convert("America/Los_Angeles").normalize()
    today = pd.Timestamp("2026-06-11", tz="America/Los_Angeles")
    avail = local_day <= today
    assert np.all(np.isfinite(w[avail])) and np.all(np.isnan(w[~avail]))
    # 13:00 Pacific = 20:00 UTC: the next day becomes available
    i2 = i + 1
    w2 = c.window(i2, T, H)[0]
    local_day2 = grid[i2 - T:i2 + H].tz_convert("America/Los_Angeles").normalize()
    assert np.all(np.isfinite(w2[local_day2 <= today + pd.DateOffset(days=1)]))
    assert np.all(np.isnan(w2[local_day2 > today + pd.DateOffset(days=1)]))
    # changing unpublished prices does not change the window
    dam2 = dam.copy()
    dam2[dam2.index >= pd.Timestamp("2026-06-12 07:00", tz="UTC")] += 1e6
    c2 = cv.day_ahead_price_covariate(dam2, grid)
    np.testing.assert_array_equal(c.window(i, T, H), c2.window(i, T, H))
    # ... and would change it once published
    assert not np.array_equal(c.window(i2, T, H)[0][np.isfinite(w2)], c2.window(i2, T, H)[0][np.isfinite(w2)])


def test_day_ahead_price_dst_change():
    grid = pd.date_range("2026-03-07 00:00", "2026-03-12 00:00", freq="1h", tz="UTC", inclusive="left")
    dam = pd.Series(np.arange(len(grid), dtype=float), index=grid)
    c = cv.day_ahead_price_covariate(dam, grid)
    i = int(grid.get_loc(pd.Timestamp("2026-03-07 21:00", tz="UTC")))  # 13:00 PST on 03-07
    w = c.window(i, 6, 48)[0]
    local = grid[i - 6:i + 48].tz_convert("America/Los_Angeles").normalize()
    ok = local <= pd.Timestamp("2026-03-08", tz="America/Los_Angeles")  # next day (the DST day) published at 13:00
    assert np.all(np.isfinite(w[ok])) and np.all(np.isnan(w[~ok]))


def test_archived_forecast_lead_rule_and_columns():
    grid = _grid(freq="1h")
    fc = pd.DataFrame({"f": np.arange(len(grid), dtype=float), "actual_leak": np.arange(len(grid), dtype=float) * 7}, index=grid)
    lead = pd.Timedelta("24h")
    c = cv.archived_forecast_covariate(fc, grid, ["f"], issue_lead=lead, label="x")
    i = 30
    w = c.window(i, 10, 48)[0]
    steps = np.arange(i - 10, i + 48)
    ok = steps <= i + 24  # valid time v is available if v - 24 h <= origin
    assert np.all(np.isfinite(w[ok])) and np.all(np.isnan(w[~ok]))
    # unavailable source values do not change the window; the actual column is not used
    fc2 = fc.copy()
    fc2.iloc[i + 25:, :] = -999.0
    fc2["actual_leak"] = 0.0
    c2 = cv.archived_forecast_covariate(fc2, grid, ["f"], issue_lead=lead, label="x")
    np.testing.assert_array_equal(c.window(i, 10, 48), c2.window(i, 10, 48))


def test_gb_forecast_builder_marks_unverified_vintage():
    grid = _grid(freq="30min")
    fc = pd.DataFrame({"f": np.arange(len(grid), dtype=float)}, index=grid)
    c = cv.archived_forecast_covariate(fc, grid, ["f"], issue_lead=pd.Timedelta("24h"), label="gb", vintage_verified=False)
    assert c.vintage_verified is False
    w = c.window(60, 20, 96)[0]  # 96 steps = 48 h: steps beyond 24 h are NaN
    assert np.isnan(w[20 + 49:]).all() and np.isfinite(w[:20 + 49]).all()


# ------------------------------------------------------------------------------------------------------ laptop builders
def _laptop_fixture():
    t0 = pd.Timestamp("2026-10-26 20:00:00", tz="UTC")
    segs = [("a", 20, "train", 16, 5), ("b", 10, "idle", 0, 0), ("c", 30, "square", 16, 0), ("d", 20, "train", 16, 4)]
    seg_t = [t0, t0 + pd.Timedelta(seconds=20), t0 + pd.Timedelta(seconds=30), t0 + pd.Timedelta(seconds=60)]
    rows = [(t, "segment_start") for t in seg_t]
    # observed checkpoints in segment a (period 5 s, duration 0.4 s): starts at 5.0, 10.4, 15.8
    for s in (5.0, 10.4, 15.8):
        rows += [(t0 + pd.Timedelta(seconds=s), "ckpt_start"), (t0 + pd.Timedelta(seconds=s + 0.4), "ckpt_end")]
    ev = pd.DataFrame({"kind": [k for _, k in rows]}, index=pd.DatetimeIndex([t for t, _ in rows])).sort_index()
    grid = pd.date_range(t0, periods=80, freq="1s")
    return segs, ev, grid


def test_laptop_training_on_uses_schedule_only():
    segs, ev, grid = _laptop_fixture()
    sched = cv.laptop_schedule_from_events(segs, ev, 0.4)
    fut = cv.laptop_training_on_future(sched, grid, 1.0, 12, 60)
    expect = np.zeros(60)
    # steps 12..: segment a until 20 (train), b 20-30 idle, c 30-60 square (on 30-90 s mod 120: on for 60 s, off 60 s -> on for the first 60 s of the segment... 60 s on/off)
    t = np.arange(12, 72)
    expect[(t < 20)] = 1.0
    expect[(t >= 30) & (t < 60)] = 1.0  # square: first minute on
    expect[(t >= 60) & (t < 72)] = 1.0  # segment d train
    np.testing.assert_allclose(fut, expect)
    # later segment starts that happen after the origin are not read: shift them by 7 s
    t = ev.index.to_series()
    later = (t > grid[12]) & (ev.kind == "segment_start")
    shifted = ev.copy()
    shifted.index = pd.DatetimeIndex(np.where(later, t + pd.Timedelta(seconds=7), t))
    sched2 = cv.laptop_schedule_from_events(segs, shifted.sort_index(), 0.4)
    np.testing.assert_allclose(cv.laptop_training_on_future(sched2, grid, 1.0, 12, 60), expect)


def test_laptop_ckpt_projection_uses_last_observed_checkpoint_only():
    segs, ev, grid = _laptop_fixture()
    sched = cv.laptop_schedule_from_events(segs, ev, 0.4)
    i = 17  # origin at 17 s: last observed checkpoint ended at 16.2 s
    fut = cv.laptop_ckpt_future(sched, grid, 1.0, i, 60)
    # next start 16.2 + 5 = 21.2 > segment a end (20 s): none in segment a; segment d (ck 4 s): first at 60 + 4 = 64.0
    starts = [64.0, 68.4, 72.8]
    exp = np.zeros(60)
    for s in starts:
        for k in range(len(exp)):
            lo, hi = i + k, i + k + 1
            exp[k] += max(0.0, min(hi, s + 0.4) - max(lo, s))
    np.testing.assert_allclose(fut, np.clip(exp, 0, 1), atol=1e-9)
    # a checkpoint logged after the origin does not move the projection
    extra = pd.DataFrame({"kind": ["ckpt_start", "ckpt_end"]},
                         index=pd.DatetimeIndex([grid[0] + pd.Timedelta(seconds=19.1), grid[0] + pd.Timedelta(seconds=19.6)]))
    sched2 = cv.laptop_schedule_from_events(segs, pd.concat([ev, extra]).sort_index(), 0.4)
    np.testing.assert_allclose(cv.laptop_ckpt_future(sched2, grid, 1.0, i, 60), fut)


def test_laptop_ckpt_projection_within_segment():
    segs, ev, grid = _laptop_fixture()
    sched = cv.laptop_schedule_from_events(segs, ev, 0.4)
    fut = cv.laptop_ckpt_future(sched, grid, 1.0, 12, 8)  # origin 12 s, last observed end 10.8 -> next at 15.8 (observed later) projected 15.8
    # projection: 10.8 + 5 = 15.8 -> bin 15 gets 0.2 s, bin 16 gets 0.2 s
    np.testing.assert_allclose(fut[3], 0.2, atol=1e-9)
    np.testing.assert_allclose(fut[4], 0.2, atol=1e-9)


# -------------------------------------------------------------------------------------------------------------- engine
def test_window_stats_match_shared_metrics():
    from abl_engine import Scale, window_stats

    rng = np.random.default_rng(0)
    B, H, T = 3, 20, 60
    ctx = np.cumsum(rng.standard_normal((B, T)), axis=1) + 50
    y = np.cumsum(rng.standard_normal((B, H)), axis=1) + 50
    q = np.sort(y[..., None] + rng.standard_normal((B, H, 21)) * 2, axis=-1)
    sc = Scale("t", T, H, ((1, 5), (6, 20)), 2, "1")
    st = window_stats(q, y, ctx, sc)
    for b in range(B):
        fc = QuantileForecast(q[b][None, None])
        full = st[b, -1]
        assert full[0] == pytest.approx(pinball(fc, y[b][None, None]))
        assert 2 * full[0] / full[1] == pytest.approx(wql(fc, y[b][None, None]))
        assert full[3] == pytest.approx(mase(fc, y[b][None, None], ctx[b][None, None], season=2))
        assert full[4] == pytest.approx(coverage(fc, y[b][None, None], 0.8))
        assert full[5] == pytest.approx(coverage(fc, y[b][None, None], 0.98))
        fc1 = QuantileForecast(q[b][None, None, :5])
        assert st[b, 0, 0] == pytest.approx(pinball(fc1, y[b][None, None, :5]))


def test_summarize_pairs_configs_and_gives_zero_delta_for_identical_stats():
    from abl_engine import block_indices, summarize

    rng = np.random.default_rng(1)
    a = np.abs(rng.standard_normal((2, 40, 2, 6))) + 1
    out = summarize({"none": a, "same": a.copy(), "worse": a * 1.5}, none_key="none", block=3, reps=200)
    d_same = out["boot"]["same"] - out["boot"]["none"]
    assert np.allclose(d_same, 0.0)
    idx = block_indices(np.random.default_rng(0), 40, 3, 5)
    assert idx.shape == (5, 40) and idx.min() >= 0 and idx.max() < 40
    assert np.all(out["point"]["worse"][..., 1] > out["point"]["none"][..., 1])


def test_guard_constant_rows():
    from abl_engine import guard_constant_rows

    cov = np.zeros((3, 4, 10))
    cov[0, 0] = 350.0  # constant, non-binary -> zeroed
    cov[0, 1] = 1.0  # constant indicator -> kept
    cov[0, 2] = np.arange(10)  # varying -> kept
    cov[0, 3, :5] = 0.08
    cov[0, 3, 5:] = np.nan  # constant over its observed part -> zeroed, NaN kept
    cov[1, 0] = [0.0, 1.0] * 5  # binary -> kept
    out, n = guard_constant_rows(cov)
    assert n == 2
    assert np.all(out[0, 0] == 0.0) and np.all(out[0, 1] == 1.0)
    np.testing.assert_array_equal(out[0, 2], np.arange(10))
    assert np.all(out[0, 3, :5] == 0.0) and np.all(np.isnan(out[0, 3, 5:]))
    np.testing.assert_array_equal(out[1, 0], cov[1, 0])
    assert cov[0, 0, 0] == 350.0  # input untouched
