import numpy as np
import pytest

from heatwave.forecast import (
    QUANTILE_LEVELS, ARIMAForecaster, ETSForecaster, PerfectForesight,
    QuantileForecast, SeasonalNaiveForecaster,
)
from heatwave.forecast.metrics import coverage, pinball, wql


def _series(n=600, s=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + np.sin(2 * np.pi * t / s) + 0.1 * rng.standard_normal(n)


def test_quantile_interp_and_sorting():
    v = np.random.default_rng(0).standard_normal((2, 1, 5, 21))
    fc = QuantileForecast(v)
    assert np.all(np.diff(fc.values, axis=-1) >= 0)
    assert fc.quantile(0.5).shape == (2, 1, 5)
    lo, hi = fc.interval(0.8)
    assert np.all(lo <= hi)
    assert np.allclose(fc.quantile(0.125), 0.5 * (fc.quantile(0.1) + fc.quantile(0.15)))


@pytest.mark.parametrize("f", [SeasonalNaiveForecaster(24), ETSForecaster(24), ARIMAForecaster()])
def test_classical_shapes_and_sanity(f):
    y = _series()
    ctx, fut = y[:-48], y[-48:]
    fc = f.forecast(ctx, 48)
    assert fc.values.shape == (1, 1, 48, len(QUANTILE_LEVELS))
    assert np.isfinite(fc.values).all()
    assert pinball(fc, fut[None, None]) < 1.0


def test_oracle_is_exact():
    y = _series()
    o = PerfectForesight()
    o.set_future(y[-48:])
    fc = o.forecast(y[:-48], 48)
    assert pinball(fc, y[None, None, -48:]) == 0.0
    assert coverage(fc, y[None, None, -48:]) == 1.0


def test_seasonal_naive_handles_nan():
    y = _series()
    y[100:110] = np.nan
    fc = SeasonalNaiveForecaster(24).forecast(y, 24)
    assert np.isfinite(fc.values).all()


def test_timestral_wrapper_smoke():
    pytest.importorskip("timestral")
    from heatwave.forecast.timestral_fc import TimestralForecaster

    try:
        f = TimestralForecaster("5M", threads=2)
    except Exception as e:  # weights unavailable offline
        pytest.skip(f"Timestral weights unavailable: {e}")
    y = _series(2048)
    kc = np.stack([np.sin(2 * np.pi * np.arange(2048 + 48) / 24)])[None]
    fc = f.forecast(y[:2000], 48, known_covariates=kc[..., :2048])
    assert fc.values.shape == (1, 1, 48, 21)
    assert wql(fc, y[None, None, 2000:2048]) < 0.2
