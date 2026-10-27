"""Checks on the processed public datasets: shapes, UTC index, units in meta, no duplicated timestamps."""

import json

import numpy as np
import pandas as pd
import pytest

from heatwave.data import public
from heatwave.data.public.weather import wet_bulb_stull
from heatwave.data.schema import PROCESSED, list_series, load_series

# MIT Supercloud series are not redistributed (CC BY-NC-ND 4.0); rebuild them with scripts/fetch_mit_supercloud.py
needs_mit = pytest.mark.skipif(not (PROCESSED / "public_gpu_mit_jobs.parquet").exists(),
                               reason="MIT Supercloud series not present; run scripts/fetch_mit_supercloud.py")

TOP = [n for n in list_series(PROCESSED) if n.startswith("public_")]
GPU_SERIES = public.list_gpu_series() if public.GPU_DIR.exists() else []
NODE_SESSIONS = list(public.list_node_power_sessions().session) if (PROCESSED / "public_gpu_node_sessions.parquet").exists() else []
NLR_RUNS = list(public.list_nlr_runs().run) if (PROCESSED / "public_gpu_nlr_runs.parquet").exists() else []


def _check_series(df, meta):
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    steps = np.unique(np.diff(df.index.asi8)).tolist()
    assert steps == [int(pd.Timedelta(meta["freq"]).value)], f"irregular index: {steps[:5]}"
    assert set(df.columns) <= set(meta["columns"])
    assert all(meta["columns"][c].get("unit") for c in df.columns)
    assert meta["data_kind"] == "public"
    assert meta["source"]["raw_sha256"], "raw sha256 missing"
    assert meta["source"]["url"] and meta["source"]["license"]


@pytest.mark.parametrize("name", TOP)
def test_top_level_series(name):
    df, meta = load_series(name)
    assert len(df) > 100
    _check_series(df, meta)


def test_expected_top_level_series_exist():
    for n in ["public_caiso_np15_lmp_dam", "public_caiso_np15_lmp_rtm", "public_caiso_carbon", "public_caiso_demand",
              "public_uk_carbon_intensity", "public_de_price_dam_1h", "public_de_generation", "public_de_forecast_dayahead",
              "public_weather_nasa_bayarea", "public_weather_nasa_london", "public_weather_nasa_frankfurt"]:
        assert n in TOP, n


@needs_mit
def test_gpu_jobs_table():
    jobs = public.load_gpu_jobs()
    assert len(jobs) >= 200
    assert (jobs.duration_s >= 1800).all()
    assert {"job_id", "n_nodes", "n_gpus", "node_ids", "start_utc", "end_utc"} <= set(jobs.columns)
    assert jobs.explode("node_ids").node_ids.nunique() >= 20
    assert str(jobs.start_utc.dt.tz) == "UTC"
    assert jobs.job_id.is_unique


@needs_mit
def test_gpu_series_all_files_consistent():
    jobs = public.load_gpu_jobs()
    listed = {s for ss in jobs.series for s in ss}
    assert listed == set(GPU_SERIES)


@needs_mit
def test_gpu_series_sample():
    jobs = public.load_gpu_jobs()
    sample = jobs.sample(12, random_state=0)
    for r in sample.itertuples():
        node = None if r.n_nodes == 1 else r.node_ids[0]
        df = public.load_gpu_series(r.job_id, node)
        meta = public.load_gpu_meta(r.job_id, node)
        _check_series(df, meta)
        assert len(df) >= 1800
        gpus = sorted({c.split("_")[0] for c in df.columns if c.startswith("g") and c[1].isdigit()})
        assert len(gpus) >= 1
        p = df[[f"{g}_power_w" for g in gpus]].sum(axis=1, min_count=len(gpus))
        both = pd.concat([p, df.node_power_w], axis=1).dropna()
        assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-4)
        assert meta["columns"][f"{gpus[0]}_power_w"]["unit"] == "W"
        assert meta["columns"][f"{gpus[0]}_temp_c"]["unit"] == "degC"
        assert df[f"{gpus[0]}_power_w"].dropna().between(0, 500).all()  # V100 board limit is 300 W; margin for sensor noise


@needs_mit
def test_multi_node_job_requires_node():
    jobs = public.load_gpu_jobs()
    multi = jobs[jobs.n_nodes > 1]
    if multi.empty:
        pytest.skip("no multi-node job collected")
    with pytest.raises(ValueError):
        public.load_gpu_series(multi.iloc[0].job_id)


def test_loaders_shapes_and_units():
    assert list(public.load_price("caiso", "dam").columns)[0] == "lmp_usd_per_mwh"
    assert public.load_meta("public_caiso_np15_lmp_dam")["columns"]["lmp_usd_per_mwh"]["unit"] == "USD/MWh"
    assert len(public.load_price("caiso", "rtm")) > 100_000
    assert public.load_price("de", "dam").columns.tolist() == ["price_eur_per_mwh"]
    for region in ("gb", "caiso", "de"):
        c = public.load_carbon(region, "actual")
        assert c.columns.tolist() == ["intensity_g_per_kwh"]
        assert c.index.tz is not None
        assert c.intensity_g_per_kwh.dropna().between(0, 1500).all()
    gb = public.load_carbon("gb", "forecast")
    assert gb.intensity_g_per_kwh.notna().mean() > 0.9
    with pytest.raises(ValueError):
        public.load_carbon("caiso", "forecast")
    for region in ("bayarea", "london", "frankfurt"):
        w = public.load_weather(region, "actual")
        assert {"temperature_2m_c", "relative_humidity_2m_pct"} <= set(w.columns)
        assert w.temperature_2m_c.dropna().between(-40, 55).all()
    assert public.load_forecast_covariates("de").shape[1] == 4
    assert "demand_forecast_day_ahead_mw" in public.load_forecast_covariates("caiso")


def test_weather_forecast_files_if_collected():
    try:
        w = public.load_weather("london", "forecast")
    except FileNotFoundError:
        pytest.skip("Open-Meteo forecast files not collected (quota)")
    assert w.index.tz is not None and not w.index.duplicated().any()


def test_wet_bulb_stull_reference_values():
    # Stull (2011) worked example: T = 20 degC, RH = 50 % gives about 13.7 degC
    assert abs(float(wet_bulb_stull(20.0, 50.0)) - 13.7) < 0.15
    assert abs(float(wet_bulb_stull(25.0, 100.0)) - 25.0) < 0.5  # at saturation wet bulb equals air temperature; formula is an approximation
    assert float(wet_bulb_stull(30.0, 20.0)) < float(wet_bulb_stull(30.0, 80.0))


def test_manifests_have_sha256():
    for f in PROCESSED.glob("public_*/raw_manifest.json"):
        m = json.loads(f.read_text())
        assert m and all(len(v["sha256"]) == 64 for v in m.values()), f


def test_job_traces():
    pods = public.load_job_trace("alibaba2023_pods")
    assert len(pods) == 8152 and {"num_gpu", "create_s", "scheduled_s", "deletion_s"} <= set(pods.columns)
    s = public.load_job_trace("alibaba2026_gpu_pods_sample")
    assert len(s) > 10_000 and (s.gpu_request > 0).all() and (s.duration_hours >= 0).all()


@pytest.mark.parametrize("session", NODE_SESSIONS)
def test_node_power_sessions(session):
    for freq, root in public.NODE_DIRS.items():
        df, meta = load_series(session, root)
        _check_series(df, meta)
        assert meta["freq"] == freq
        assert meta["columns"]["g0_power_w"]["unit"] == "W"
        assert df.node_power_w.notna().mean() > 0.9
    d1 = public.load_node_power(session, "1s")
    assert len(d1) in (900, 901) and meta["session"]["n_gpus"] == 8
    assert meta["columns"]["g0_temp_c"]["unit"] == "degC"
    assert d1.filter(regex=r"^g\d_power_w$").max().max() < 1500  # B200 limit 1000 W; H100 700 W; short transients allowed


def test_node_sessions_table():
    t = public.list_node_power_sessions()
    assert len(t) == 32 and set(t.hardware) == {"H100", "B200"} and t.raw_content_sha256.is_unique


@pytest.mark.parametrize("run", NLR_RUNS)
def test_nlr_runs(run):
    df, meta = load_series(run, public.NLR_DIR)
    _check_series(df, meta)
    assert meta["columns"]["total_power_w"]["unit"] == "W"
    assert meta["time_origin"].startswith("relative")
    nodes = meta["run"]["nodes"]
    if nodes:  # 4 H100 SXM (700 W each, NVIDIA datasheet) + 2 CPUs per node: 5 kW per node is a loose ceiling
        assert df.total_power_w.dropna().between(0, 5000 * nodes).all()
    else:
        assert df.total_power_w.dropna().between(0, 20_000).all()
    one = public.load_nlr_run(run, "1s")
    assert len(one) <= len(df)


def test_nlr_run_counts():
    t = public.list_nlr_runs()
    assert (t.kind == "training").sum() == 41 and (t.kind == "inference_online").sum() > 0
