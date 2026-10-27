"""Zero-shot versus fine-tuned Timestral on the held-out (TEST) windows.

Windows: the evaluation windows of the multi-scale forecasting study (``heatwave.forecast.datasets``: same
splits, strides and per-series caps, same context / horizon per scale) for the measured laptop run2, the public GPU
telemetry TEST sessions / runs / jobs and the grid series from 2026-05-01, plus the simulator TEST seeds 100-109 run on
the PUBLIC environment (sim-scenario-v3; windows cut with the same rule as the study, covariates from
``heatwave.finetune.sim.SimCovariates``). Scores are the study's: WQL (ratio of means), MASE of the median, 80 % and
98 % coverage, horizon buckets, 95 % percentile bootstrap over windows and over groups
(``experiments/forecast/_common.py``).

Modes per dataset: ``none`` (no covariates) and, where the training recipe has covariates for the task, ``cov_all``
(all groups of that task).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import sim as S
from .data import ALL_GROUPS, TASKS, Recipe, guard_constant_rows

ROOT = Path(__file__).resolve().parents[2]
REAL_DATASETS = ("laptop_power", "elsayed_gpu_power", "nlr_job_power", "mit_node_power", "laptop_temp", "mit_node_temp",
                 "caiso_rt_price_1h", "uk_carbon_actual_1h")
SIM_DATASETS = {"sim_rack_power_1s": "rack_power_1s", "sim_rack_temp_1min": "rack_temp_1min", "sim_site_power_1h": "site_power_1h"}
COV_DATASETS = {"laptop_power": ("L",), "sim_rack_power_1s": S.GROUPS, "sim_rack_temp_1min": S.GROUPS, "sim_site_power_1h": S.GROUPS}


def _study_common():
    """experiments/forecast/_common.py of this repo (scoring and bootstrap shared with the forecasting study)."""
    name = "hw_forecast_common"
    if name in sys.modules:
        return sys.modules[name]
    p = ROOT / "experiments" / "forecast" / "_common.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class EvalSet:
    name: str
    scale: str
    data_kind: str
    windows: object                       # heatwave.forecast.datasets.Windows
    cov: Optional[dict] = None            # {group: (N, C, T + H)}
    groups: tuple = ()

    @property
    def n(self) -> int:
        return self.windows.n


def _sim_eval_set(name: str, kind: str, recipe: Recipe, cache_dir: Path, spec, log) -> EvalSet:
    from heatwave.forecast import datasets as D

    sk = S.SIM_KINDS[kind]
    scenario = S.load_scenario()
    fx = S.usd_per_eur()[0]
    series = []
    unit_of = {}
    scale = {"rack_power_1s": "seconds", "rack_temp_1min": "minutes", "site_power_1h": "hours"}[kind]
    for sd in recipe.test_seeds:
        S.run_sim(sd, cache_dir, fx)
        z = S.load_sim(sd, cache_dir)
        units = [str(u) for u in (z["rack_ids"] if sk["level"] == "rack" else z["site_ids"])]
        t0 = S.seed_start_utc(sd, scenario)
        for ui, u in enumerate(units):
            sid = f"seed{sd}/{u}"
            unit_of[sid] = (sd, ui)
            series.append(D.Series(id=sid, values=z[sk["key"]][:, ui].astype(np.float64), step_s=sk["step_s"], t0=t0, group=f"seed{sd}"))
    w = D.make_windows(name, scale, "simulated", series, stride=spec.stride, max_per_series=spec.max_per_series,
                       meta={"description": f"SIMULATED {kind} (sim-scenario-v3 on the public environment), TEST seeds {list(recipe.test_seeds)}"})
    T, H = w.context.shape[1], w.future.shape[1]
    covs = {g: [] for g in S.GROUPS}
    builders: dict = {}
    memo: dict = {}
    for i in range(w.n):
        sd, ui = unit_of[str(w.series_id[i])]
        o = int(w.origin_idx[i])
        if sd not in builders:
            builders[sd] = S.SimCovariates(sd, scenario, fx)
        key = (sd, o)
        if key not in memo:
            if len(memo) > 64:
                memo.clear()
            memo[key] = builders[sd].window(sk["level"], sk["step_s"], o, T, H)
        for g in S.GROUPS:
            covs[g].append(memo[key][g][:, ui, :])
    cov = {g: np.stack(v).astype(np.float64) for g, v in covs.items()}
    log(f"eval set {name}: {w.n} windows, {len(set(w.group))} groups")
    return EvalSet(name, scale, "simulated", w, cov, S.GROUPS)


def build_eval_sets(recipe: Recipe, cache_dir: str | Path, log: Callable = print, only: Optional[set] = None) -> list:
    from heatwave.forecast import datasets as D

    cache_dir = Path(cache_dir)
    specs = D.dataset_specs(sim_seeds=recipe.test_seeds, mit=recipe.use_mit)
    out = []
    for name in REAL_DATASETS:
        if name not in specs or (only and name not in only):
            continue
        spec = specs[name]
        if name in ("laptop_power", "laptop_temp"):
            q = "power" if name == "laptop_power" else "temp"
            s = D.laptop_series("run2", q, True)
            for x in s:
                x.cov, x.cov_names = x.cov[:1], ("training_on",)        # scripted schedule only; ckpt_scheduled is not used
            w = D.make_windows(spec.name, spec.scale, spec.data_kind, s, stride=spec.stride, max_per_series=spec.max_per_series,
                               meta={"quantity": spec.quantity, "unit": spec.unit, "description": spec.description})
            cov = {"L": w.known_cov.astype(np.float64)} if w.known_cov is not None else None
            out.append(EvalSet(name, spec.scale, spec.data_kind, w, cov if name in COV_DATASETS else None,
                               ("L",) if name in COV_DATASETS else ()))
        else:
            w = D.build_dataset(spec)
            out.append(EvalSet(name, spec.scale, spec.data_kind, w))
        log(f"eval set {name}: {out[-1].n} windows")
    for name, kind in SIM_DATASETS.items():
        if only and name not in only:
            continue
        spec = specs[name]
        out.append(_sim_eval_set(name, kind, recipe, cache_dir, spec, log))
    return out


# ------------------------------------------------------------------------------------------------ forecasting and scoring
def _known(es: EvalSet, mode: str, idx: np.ndarray) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    parts = [es.cov[g][idx] for g in ALL_GROUPS if g in es.groups]
    known = np.concatenate(parts, axis=1)
    known, _ = guard_constant_rows(known)
    return known


def forecast_set(f, es: EvalSet, mode: str, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Quantile forecasts ``(N, 1, H, 21)`` and a validity mask ``(N,)`` (windows the model rejects are NaN)."""
    w = es.windows
    N, H = w.n, w.future.shape[1]
    vals = np.full((N, 1, H, 21), np.nan, dtype=np.float32)
    for i in range(0, N, batch):
        idx = np.arange(i, min(N, i + batch))
        known = _known(es, mode, idx)
        try:
            vals[idx] = f.forecast(w.context[idx][:, None, :], H, known_covariates=known).values.astype(np.float32)
        except ValueError:
            for j in idx:
                try:
                    kj = None if known is None else known[j - i:j - i + 1]
                    vals[j] = f.forecast(w.context[j:j + 1][:, None, :], H, known_covariates=kj).values.astype(np.float32)
                except ValueError:
                    pass
    ok = np.isfinite(vals).all(axis=(1, 2, 3))
    return vals, ok


def evaluate_models(models: dict, eval_sets: list, *, n_boot: int = 1000, log: Callable = print, reference: Optional[str] = None,
                    batch: int = 64) -> dict:
    """Score every model on every eval set and mode. ``models``: label -> forecaster with ``.forecast`` (the heatwave
    wrapper). ``reference`` (default: the first label) is the zero-shot model that deltas are taken against."""
    C = _study_common()
    from heatwave.forecast import datasets as D

    reference = reference or next(iter(models))
    out: dict = {"datasets": {}, "macro": {}}
    for es in eval_sets:
        w = es.windows
        modes = ["none"] + (["cov_all"] if es.cov is not None and es.groups else [])
        buckets = D.BUCKETS[es.scale]
        info = {"scale": es.scale, "data_kind": es.data_kind, "n_windows": int(w.n), "n_groups": int(len(set(w.group))),
                "context": int(w.context.shape[1]), "horizon": int(w.future.shape[1]), "step_s": float(w.step_s),
                "season": int(w.season), "covariate_groups": list(es.groups), "meta": w.meta, "modes": {}}
        for mode in modes:
            t0 = time.time()
            vals, ok = {}, np.ones(w.n, dtype=bool)
            for label, f in models.items():
                vals[label], okm = forecast_set(f, es, mode, batch)
                ok &= okm
            keep = np.nonzero(ok)[0]
            comps = {}
            for label in models:
                comps[label] = C.score_windows(vals[label][keep], w.future[keep], w.context[keep], w.season, buckets)
            grp = w.group[keep]
            block = {"n_scored": int(len(keep)), "n_dropped_invalid": int(w.n - len(keep)), "models": {}, "delta_vs_reference": {}}
            for label in models:
                block["models"][label] = {b: C.bootstrap_summary(c, grp, n_boot) for b, c in comps[label].items()}
                if label != reference:
                    block["delta_vs_reference"][label] = {b: C.bootstrap_delta(comps[label][b], comps[reference][b], grp, n_boot)
                                                          for b in comps[label]}
            info["modes"][mode] = block
            msg = " ".join(f"{lb}={block['models'][lb]['all']['wql']['est']:.4f}" for lb in models)
            log(f"{es.name} [{mode}] n={len(keep)} WQL(all): {msg} ({time.time() - t0:.0f}s)")
        out["datasets"][es.name] = info
    for mode in ("none", "cov_all"):
        names = [n for n, d in out["datasets"].items() if mode in d["modes"]]
        if not names:
            continue
        m = {}
        for label in models:
            wq = [out["datasets"][n]["modes"][mode]["models"][label]["all"]["wql"]["est"] for n in names]
            m[label] = {"datasets": names, "mean_wql": float(np.mean(wq))}
            if label != reference:
                ref = [out["datasets"][n]["modes"][mode]["models"][reference]["all"]["wql"]["est"] for n in names]
                m[label]["mean_wql_ratio_to_reference"] = float(np.mean(np.asarray(wq) / np.asarray(ref)))
                m[label]["n_datasets_lower_wql"] = int(np.sum(np.asarray(wq) < np.asarray(ref)))
        out["macro"][mode] = m
    out["reference"] = reference
    return out
