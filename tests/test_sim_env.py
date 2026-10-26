import numpy as np
import pytest


def test_public_environment_drives_the_simulator():
    hs = pytest.importorskip("gpu_stack.research.heatwave_sim")
    from heatwave.sim_env import public_environment

    pe = public_environment("2026-06-01T00:00:00Z", 6 * 3600, usd_per_eur=1.1)
    assert set(pe.environment) == {"site-a", "site-b"}
    a = pe.environment["site-a"]
    assert a.outdoor_c.size == 7 and np.isfinite(a.price_usd_per_kwh).all()
    # carbon forecast is the value 24 h earlier: it must not equal the actual series
    f = pe.forecasts["site-a"]["carbon_forecast_g_per_kwh"]
    assert f.size == 7 and not np.allclose(f, a.carbon_g_per_kwh)
    from pathlib import Path
    import gpu_stack

    path = Path(gpu_stack.__file__).resolve().parents[1] / "experiments/e003-heatwave/sim-scenario-v2.json"
    if not path.exists():
        pytest.skip("gpu_stack checkout without e003 scenarios")
    sc = hs.load_scenario(path)
    sc["horizon_s"] = 6 * 3600
    sc["outputs"] = {"gpu_windows": [[0, 60]]}
    res = hs.run_scenario(sc, seed=0, environment=pe.environment)
    arr = res.to_arrays()
    assert np.isfinite(arr["site_power_w"]).all()
