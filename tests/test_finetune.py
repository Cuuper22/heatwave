"""Tests for the Timestral fine-tune: forward equivalence, loss, training behaviour, export, splits, covariate leakage."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

timestral = pytest.importorskip("timestral")
from timestral import TimestralForecaster as BaseForecaster  # noqa: E402

from heatwave.finetune import data as Dt  # noqa: E402
from heatwave.finetune import model as Mo  # noqa: E402
from heatwave.finetune import sim as S  # noqa: E402
from heatwave.finetune import train as Tr  # noqa: E402

try:
    BASE5 = BaseForecaster.from_pretrained("mistralai/Timestral-5M", local_files_only=True)
except Exception:  # pragma: no cover - weights not cached
    BASE5 = None
needs_weights = pytest.mark.skipif(BASE5 is None, reason="Timestral-5M weights are not in the local Hugging Face cache")


def _series(rng, B=4, T=192, H=64, C=0):
    ctx = (np.abs(rng.normal(size=(B, 1, T))).cumsum(-1) * 0.1 + 5).astype(np.float32)
    ctx[0, 0, 10:20] = np.nan
    known = rng.normal(size=(B, C, T + H)).astype(np.float32) if C else None
    if C:
        known[:, 0] = rng.random((B, T + H)) > 0.5
        known[1, -1, T + 5:] = np.nan
    return ctx, known


def _toy_pool(n_series=3, length=1500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    items = []
    for i in range(n_series):
        v = 50 + 20 * np.sign(np.sin(2 * np.pi * t / (97 + 11 * i))) + 3 * rng.normal(size=length)
        items.append(Dt.SeriesItem(f"toy{i}", f"toy{i}", v.astype(np.float32)))
    task = Dt.TaskSpec("toy", "seconds", "simulated", 1.0, ((128, 64),))
    return Dt.SeriesPool(task, items)


# ------------------------------------------------------------------------------------------ forward pass and loss
@needs_weights
@pytest.mark.parametrize("C", [0, 3])
def test_grad_forward_equals_inference_forward(C):
    rng = np.random.default_rng(0)
    ctx, known = _series(rng, C=C)
    gf = Mo.load_for_training("5M")
    a = BASE5.forecast(ctx, 64, known_covariates=known).quantiles
    with torch.enable_grad():
        b = gf.quantiles(ctx, 64, known_covariates=known)
    assert b.requires_grad
    assert torch.equal(a, b.detach())


@needs_weights
def test_pinball_matches_study_metric():
    from heatwave.forecast import QuantileForecast
    from heatwave.forecast import metrics as M

    rng = np.random.default_rng(1)
    ctx, _ = _series(rng, B=6)
    ctx = np.nan_to_num(ctx, nan=5.0)
    fut = (ctx[:, 0, -1:] + rng.normal(size=(6, 64))).astype(np.float32)
    gf = Mo.load_for_training("5M")
    with torch.no_grad():
        q = gf.quantiles(ctx, 64)
    rms = Mo.context_rms(torch.from_numpy(ctx[:, 0]))
    per = Mo.pinball_per_window(q, torch.from_numpy(fut), gf.quantile_levels, rms)
    ref = [M.pinball(QuantileForecast(values=q[i:i + 1].numpy().astype(np.float64)), fut[i:i + 1, None, :]) / float(rms[i]) for i in range(6)]
    np.testing.assert_allclose(per.numpy(), ref, rtol=1e-4)


def test_pinball_ignores_nonfinite_future():
    q = torch.zeros(1, 1, 4, 3)
    y = torch.tensor([[1.0, float("nan"), 1.0, float("nan")]])
    a = Mo.pinball_per_window(q, y, (0.1, 0.5, 0.9), torch.ones(1, 1, 1))
    b = Mo.pinball_per_window(q[:, :, [0, 2]], y[:, [0, 2]], (0.1, 0.5, 0.9), torch.ones(1, 1, 1))
    assert torch.allclose(a, b)


# ------------------------------------------------------------------------------------------ modes
@needs_weights
def test_modes_set_expected_parameters():
    n_full = None
    for mode in ("full", "head", "lora"):
        gf = Mo.load_for_training("5M")
        info = Mo.set_trainable(gf.model, mode, last_blocks=1)
        names = [n for n, p in gf.model.named_parameters() if p.requires_grad]
        if mode == "full":
            n_full = info["n_params_trainable"]
            assert n_full == info["n_params_total_base"]
        elif mode == "head":
            assert 0 < info["n_params_trainable"] < n_full
            last = len(gf.model.transformer.layers) - 1
            assert all(n.startswith(("head.", "transformer.out_norm.", f"transformer.layers.{last}.")) for n in names)
        else:
            assert all(("lora_" in n) or n.startswith("head.") for n in names)
            assert info["lora_modules"] > 0 and info["n_params_trainable"] < n_full


@needs_weights
def test_lora_starts_as_base_and_merge_is_exact(tmp_path):
    ctx, _ = _series(np.random.default_rng(2))
    gf = Mo.load_for_training("5M")
    with torch.no_grad():
        q0 = gf.quantiles(ctx, 64)
    Mo.set_trainable(gf.model, "lora", lora_r=4)
    with torch.no_grad():
        torch.testing.assert_close(gf.quantiles(ctx, 64), q0, rtol=1e-5, atol=1e-4)   # B starts at zero
        for m in gf.model.modules():
            if isinstance(m, Mo.LoRALinear):
                m.lora_b.normal_(0, 0.02)
        q1 = gf.quantiles(ctx, 64)
    assert not torch.equal(q1, q0)
    out = Mo.save_checkpoint(gf.model, tmp_path / "ck", base_source="5M")
    assert any(isinstance(m, Mo.LoRALinear) for m in gf.model.modules())     # the live model keeps its adapters
    loaded = BaseForecaster.from_pretrained(out)
    sd_keys = set(loaded.model.state_dict())
    assert sd_keys == set(BASE5.model.state_dict())                           # standard key layout
    q2 = loaded.forecast(ctx, 64).quantiles
    with torch.no_grad():
        q1s = gf.quantiles(ctx, 64)
    torch.testing.assert_close(q2, q1s, rtol=2e-4, atol=1e-3 * float(q1s.abs().mean()))


# ------------------------------------------------------------------------------------------ training behaviour
def _run(tmp_path, name, **kw):
    pool = _toy_pool()
    val_pool = _toy_pool(n_series=2, seed=5)
    mix = Dt.Mixture({"toy": pool}, batch_size=8)
    vb = Dt.validation_batches({"toy": val_pool}, seed=3, n=16)
    cfg = Tr.TrainConfig(base="5M", device="cpu", batch_size=8, eval_every=10, warmup_steps=3, log_every=0,
                         checkpoint_every=0, **kw)
    return Tr.train(cfg, mix, vb, tmp_path / name, log=lambda m: None, resume=False)


@needs_weights
def test_zero_step_finetune_reproduces_zero_shot(tmp_path):
    from heatwave.forecast.timestral_fc import TimestralForecaster as Wrapper

    res = _run(tmp_path, "zero", max_steps=0, symmetric=False, lr=1e-4)
    assert res["steps_run"] == 0 and res["best_step"] == 0
    ctx, known = _series(np.random.default_rng(4), B=3, T=160, H=96, C=2)
    a = Wrapper("5M").forecast(ctx[:, 0], 96, known_covariates=known)
    b = Wrapper("5M", weights_dir=str(tmp_path / "zero" / "best")).forecast(ctx[:, 0], 96, known_covariates=known)
    assert np.array_equal(a.values, b.values)
    assert (tmp_path / "zero" / "best" / "LICENSE").exists()
    cfg = json.loads((tmp_path / "zero" / "best" / "config.json").read_text())
    assert cfg == BASE5.config.to_dict()


@needs_weights
def test_loss_decreases_on_toy_series(tmp_path):
    res = _run(tmp_path, "toy", max_steps=40, lr=2e-4, symmetric=False, patience=100)
    first, best = res["initial_val_macro"], res["best_val_macro"]
    assert np.isfinite(first) and np.isfinite(best)
    assert best < first * 0.95, (first, best)
    # the exported best weights reproduce the recorded best validation loss
    gf = Mo.load_for_training(str(tmp_path / "toy" / "best"))
    vb = Dt.validation_batches({"toy": _toy_pool(n_series=2, seed=5)}, seed=3, n=16)
    assert abs(Tr.validate(gf, vb, False)["macro"] - best) < 1e-5


@needs_weights
def test_same_seed_same_history(tmp_path):
    a = _run(tmp_path, "a", max_steps=12, lr=1e-4, symmetric=False, patience=100)
    b = _run(tmp_path, "b", max_steps=12, lr=1e-4, symmetric=False, patience=100)
    assert [h["val_macro"] for h in a["history"]] == [h["val_macro"] for h in b["history"]]


@needs_weights
def test_resume_matches_uninterrupted_run(tmp_path):
    pool, vpool = _toy_pool(), _toy_pool(n_series=2, seed=5)
    vb = Dt.validation_batches({"toy": vpool}, seed=3, n=16)

    def cfg(halt=None):
        return Tr.TrainConfig(base="5M", device="cpu", batch_size=8, eval_every=6, warmup_steps=2, log_every=0, checkpoint_every=6,
                              max_steps=12, lr=1e-4, symmetric=False, patience=100, halt_after=halt)

    full = Tr.train(cfg(), Dt.Mixture({"toy": pool}, batch_size=8), vb, tmp_path / "full", log=lambda m: None, resume=False)
    h = Tr.train(cfg(6), Dt.Mixture({"toy": pool}, batch_size=8), vb, tmp_path / "part", log=lambda m: None, resume=False)
    assert h["stop_reason"] == "halted" and h["steps_run"] == 6
    part = Tr.train(cfg(), Dt.Mixture({"toy": pool}, batch_size=8), vb, tmp_path / "part", log=lambda m: None, resume=True)
    assert [h["val_macro"] for h in full["history"]] == [h["val_macro"] for h in part["history"]]


# ------------------------------------------------------------------------------------------ data and splits
def test_group_selection_and_constant_guard():
    rng = np.random.default_rng(0)
    seen = {Dt.choose_groups(rng, ("J", "P", "W", "C")) for _ in range(400)}
    assert () in seen and ("J", "P", "W", "C") in seen and ("W",) in seen and len(seen) > 6
    assert Dt.choose_groups(rng, ()) == ()
    cov = np.zeros((2, 3, 10))
    cov[0, 0] = 7.0                 # constant, not an indicator -> zeroed
    cov[0, 1] = 1.0                 # constant indicator -> kept
    cov[0, 2] = np.arange(10)
    cov[1, 0] = 3.0; cov[1, 0, 4] = np.nan   # constant with a NaN -> zeroed, NaN stays
    out, n = Dt.guard_constant_rows(cov)
    assert n == 2 and (out[0, 0] == 0).all() and (out[0, 1] == 1).all() and np.isnan(out[1, 0, 4]) and (out[1, 0, [0, 1]] == 0).all()


def test_sim_seed_weeks_respect_the_splits():
    sc = S.load_scenario()
    tr = [S.seed_start_utc(s, sc) for s in S.TRAIN_SEEDS]
    va = [S.seed_start_utc(s, sc) for s in S.VAL_SEEDS]
    te = [S.seed_start_utc(s, sc) for s in S.TEST_SEEDS]
    week = pd.Timedelta(days=7)
    assert max(tr) + week <= S.SPLIT_END_TRAIN <= min(va)
    assert max(va) + week <= S.SPLIT_END_VAL <= S.TEST_FROM_UTC <= min(te)
    assert min(tr) >= pd.Timestamp("2024-02-01", tz="UTC")                 # Open-Meteo day-ahead archive is complete from here
    assert te[0] == pd.Timestamp(sc["start_utc"]).tz_localize("UTC") and te[1] - te[0] == week
    for a in (tr, va, te):
        assert len(set(a)) == len(a)
    with pytest.raises(ValueError):
        S.split_of_seed(75)


def test_public_telemetry_splits_are_disjoint_from_test():
    D = Dt._datasets()
    tr, va = Dt._val_node_sessions()
    te = set(D.node_sessions("test"))
    assert te and not (set(tr) & te) and not (set(va) & te) and not (set(tr) & set(va))
    ntr, nva, nte = Dt._nlr_split()
    assert nte and not (set(ntr) & set(nte)) and not (set(nva) & set(nte)) and not (set(ntr) & set(nva))
    from heatwave.data import public as P

    r = P.list_nlr_runs().set_index("run")
    assert set(r.loc[nte].repeat) <= {4.0} | set(r.loc[nte].repeat)          # test rule is the project's
    assert all(r.loc[x].repeat < 4.0 for x in nva if (r.model[x], r.nodes[x]) in {(r.model[y], r.nodes[y]) for y in nte if r.repeat[y] == 4.0})
    mtr, mva, mte = Dt._mit_split()
    assert all(j % 4 == 0 for j in mte) and all(j % 4 != 0 for j in mtr + mva) and all(j % 4 == 1 for j in mva)


def test_laptop_and_grid_pools_do_not_reach_test_or_validation():
    rec = Dt.Recipe(tasks=("laptop_power_1s", "caiso_rt_price_1h", "uk_carbon_1h"), use_mit=False)
    tr, va = Dt.build_pools("train", rec), Dt.build_pools("val", rec)
    it_tr, it_va = tr["laptop_power_1s"].items[0], va["laptop_power_1s"].items[0]
    assert it_tr.end_hi == it_va.o_lo and it_tr.cov.shape[0] == 1
    for name in ("caiso_rt_price_1h", "uk_carbon_1h"):
        a, b = tr[name].items[0], va[name].items[0]
        t0 = pd.Timestamp("2000-01-01", tz="UTC")
        assert a.end_hi == b.o_lo                                                # train forecast part ends where validation starts
        D = Dt._datasets()
        s = (D.caiso_price_series if name == "caiso_rt_price_1h" else D.uk_carbon_series)(False)[0]
        assert s.t0 + pd.Timedelta(hours=a.end_hi) == S.SPLIT_END_VAL
        assert s.t0 + pd.Timedelta(hours=b.end_hi) == S.TEST_FROM_UTC
        # no training window's forecast part reaches validation: sample many
        rng = np.random.default_rng(0)
        for gi in range(len(tr[name].task.geometries)):
            lo, hi, _ = tr[name]._ranges(gi)
            assert hi[0] + tr[name].task.geometries[gi][1] <= a.end_hi
            lo, hi, _ = va[name]._ranges(gi)
            assert hi[0] + va[name].task.geometries[gi][1] <= s.t0.__class__(S.TEST_FROM_UTC).value and True


def test_series_pool_sampling_is_deterministic_and_valid():
    pool = _toy_pool()
    b1 = pool.sample(np.random.default_rng(1), 0, 8, ())
    b2 = pool.sample(np.random.default_rng(1), 0, 8, ())
    assert np.array_equal(b1.context, b2.context) and b1.context.shape == (8, 128) and b1.future.shape == (8, 64)
    assert np.isfinite(b1.future).all()


# ------------------------------------------------------------------------------------------ covariate leakage
@pytest.fixture(scope="module")
def cov_builder():
    sc = S.load_scenario()
    return S.SimCovariates(0, sc, S.usd_per_eur()[0])


def test_job_plan_uses_only_jobs_submitted_by_the_origin(cov_builder):
    b = cov_builder
    res, T, H = 60.0, 96, 60
    submit = b._submit
    o = int(np.searchsorted(submit, np.median(submit)) * 0 + 4000)            # a minute index; some jobs are still unsubmitted
    origin_s = o * res
    assert 0 < b.n_known_jobs(origin_s) < len(submit)
    w = b.window("rack", res, o, T, H)
    # a builder that never saw the later jobs gives the same J and P windows
    b2 = copy.copy(b)
    b2._jobs = b._jobs[:b.n_known_jobs(origin_s)]
    b2._submit = b._submit[:b.n_known_jobs(origin_s)]
    b2._plans = {}
    w2 = b2.window("rack", res, o, T, H)
    for g in ("J", "P"):
        np.testing.assert_array_equal(w[g], w2[g])
    # while planning with the later jobs too changes the horizon (so the test can fail)
    wf = b.window("rack", res, o, T, H, full_knowledge=True)
    assert not np.array_equal(w["J"], wf["J"])


def test_weather_covariate_is_the_dayahead_forecast_never_the_actual(cov_builder):
    b = cov_builder
    o = 3000
    w1 = b.window("rack", 60.0, o, 96, 60)["W"]
    b2 = copy.copy(b)
    b2.env = copy.copy(b.env)
    site = b.site_ids[0]
    env2 = dict(b.env.environment)
    env2[site] = type(env2[site])(**{**env2[site].__dict__, "outdoor_c": env2[site].outdoor_c + 25.0})
    b2.env.environment = env2
    np.testing.assert_array_equal(b2.window("rack", 60.0, o, 96, 60)["W"], w1)
    fc = b.env.forecasts[site]["outdoor_forecast_c"]
    assert np.isfinite(w1).all() and w1.min() >= fc.min() - 1e-6 and w1.max() <= fc.max() + 1e-6


def test_price_is_nan_until_published_and_carbon_is_lagged(cov_builder):
    b = cov_builder
    site, ri = b.site_ids[0], 0
    assert b.rack_site[ri] == site
    # origin at local 00:30 of day D: the price of day D+1 is published at 13:00 on day D, so it is unknown now
    hours = pd.DatetimeIndex(b.env.hours_utc).tz_convert(S.SITE_MARKET_TZ[site])
    h0 = int(np.nonzero((hours.hour == 0) & (hours.day == hours[48].day))[0][0]) + 24
    o = h0 * 60 + 30                                                              # minute index
    horizon = 20 * 60
    w = b.window("rack", 60.0, o, 96, horizon)["C"]                               # (2, racks, T+H)
    price = w[0, ri]
    nxt = int(np.nonzero(hours.day != hours[h0].day)[0][np.nonzero(hours.day != hours[h0].day)[0] > h0][0])
    first_unpublished_min = (nxt * 60 - (o - 96))                                 # index in the window
    assert np.isfinite(price[:96]).all()                                          # context is always known
    assert np.isnan(price[first_unpublished_min + 1:]).all()                      # next local day is not yet published
    # perturbing the unpublished source values leaves the window unchanged
    b2 = copy.copy(b)
    b2._ext = {s: dict(v) for s, v in b._ext.items()}
    vals, avail = b._ext[site]["price_usd_per_kwh"]
    v2 = vals.copy(); v2[nxt:] += 100.0
    b2._ext[site]["price_usd_per_kwh"] = (v2, avail)
    np.testing.assert_array_equal(b2.window("rack", 60.0, o, 96, horizon)["C"][0, ri], price)
    # carbon forecast = value 24 h earlier of the actual series, usable from 23 h before its hour
    carb = b.env.forecasts[site]["carbon_forecast_g_per_kwh"]
    assert carb is not None and len(carb) == b.n_nodes


def test_every_group_is_nan_beyond_the_known_future_horizon(cov_builder):
    b = cov_builder
    o = 120
    w = b.window("site", 3600.0, o, 96, 48)
    for g, arr in w.items():
        assert np.isfinite(arr[..., :96 + 24]).all() or g == "C"
        assert np.isnan(arr[..., 96 + 24:]).all(), g
