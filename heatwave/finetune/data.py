"""Training, validation and held-out windows for the Timestral fine-tune.

Training tasks (each: scale, data kind, geometries (context, horizon), covariate groups):

  sim_rack_power_1s     seconds  simulated plant, public environment  (256,300) (2048,256)  J P W C
  sim_rack_temp_1min    minutes  simulated plant, public environment  (96,60) (2880,240)    J P W C
  sim_site_power_1h     hours    simulated plant, public environment  (96,48) (72,24)       J P W C
  elsayed_gpu_power_1s  seconds  public (Elsayed et al. 2026, H100/B200 nodes)  (256,300)
  nlr_job_power_1s      seconds  public (NLR 2026 H100 multi-node jobs)         (256,300) (2048,256)
  mit_node_power_1s     seconds  public (MIT Supercloud V100 nodes, local copy)  (256,300) (2048,256)
  mit_node_temp_1min    minutes  public (MIT Supercloud, node max GPU temperature, 1-min means)  (96,60)
  laptop_power_1s       seconds  measured (RTX 3060 Laptop GPU, run1)           (256,300)       L
  caiso_rt_price_1h     hours    public (CAISO NP15 real-time LMP, hourly mean) (96,48)
  uk_carbon_1h          hours    public (GB carbon intensity, hourly mean)      (96,48)

Fixed splits (README of the lead addendum; the fine-tune never sees a test part):
  simulator   train seeds 0-49, validation 50-59, TEST 100-109 (weeks in heatwave.finetune.sim)
  Elsayed     TEST sessions: index (sorted by name) divisible by 3; validation: every 4th remaining session (index 1 of 4)
  NLR         TEST: repeat 4 (last repeat if a group has no repeat 4); validation: the repeat just below it
  MIT         TEST: job id mod 4 == 0; validation: job id mod 4 == 1
  laptop      run1 only (run2 is TEST): first 75 % of run1 train, the rest validation (forecast part after the split)
  grid        train: forecast part before 2026-04-01; validation: April 2026; TEST: origins from 2026-05-01 (never read here)

Covariate groups: J P W C as in heatwave.finetune.sim (known-future, leakage-safe), L the laptop ``training_on`` schedule.
Not used: the laptop ``ckpt_scheduled`` column (it is built from the logged checkpoint events of the run, not from the
schedule), the simulator's ``cov_*_carbon/price`` exports (actual series), and all test parts.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import sim as S

ALL_GROUPS = ("J", "P", "W", "C", "L")
GROUP_SIZES = {"J": 2, "P": 1, "W": 1, "C": 2, "L": 1}
MAX_CONTEXT_NAN_FRAC = 0.1   # same admission rule as the evaluation windows (heatwave.forecast.datasets.make_windows)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    scale: str
    data_kind: str
    step_s: float
    geometries: tuple
    groups: tuple = ()
    geom_probs: tuple = ()
    description: str = ""

    def probs(self) -> np.ndarray:
        p = np.asarray(self.geom_probs if self.geom_probs else [1.0] * len(self.geometries), dtype=float)
        return p / p.sum()


S_ = (256, 300)
M_ = (96, 60)
H_ = (96, 48)
TASKS: dict[str, TaskSpec] = {t.name: t for t in [
    TaskSpec("sim_rack_power_1s", "seconds", "simulated", 1.0, (S_, (2048, 256)), ("J", "P", "W", "C"), (0.7, 0.3),
             "SIMULATED rack power (public environment)"),
    TaskSpec("sim_rack_temp_1min", "minutes", "simulated", 60.0, (M_, (2880, 240)), ("J", "P", "W", "C"), (0.7, 0.3),
             "SIMULATED rack air temperature (public environment)"),
    TaskSpec("sim_site_power_1h", "hours", "simulated", 3600.0, (H_, (72, 24)), ("J", "P", "W", "C"), (0.6, 0.4),
             "SIMULATED site mean power, hourly (public environment)"),
    TaskSpec("elsayed_gpu_power_1s", "seconds", "public", 1.0, (S_,), (), (), "Elsayed et al. 2026 per-GPU power, 1 s"),
    TaskSpec("nlr_job_power_1s", "seconds", "public", 1.0, (S_, (2048, 256)), (), (0.7, 0.3), "NLR 2026 H100 job power, 1 s"),
    TaskSpec("mit_node_power_1s", "seconds", "public", 1.0, (S_, (2048, 256)), (), (0.7, 0.3), "MIT Supercloud V100 node power, 1 s"),
    TaskSpec("mit_node_temp_1min", "minutes", "public", 60.0, (M_,), (), (), "MIT Supercloud node max GPU temperature, 1-min means"),
    TaskSpec("laptop_power_1s", "seconds", "measured", 1.0, (S_,), ("L",), (), "measured laptop GPU power (energy counter), run1"),
    TaskSpec("caiso_rt_price_1h", "hours", "public", 3600.0, (H_,), (), (), "CAISO NP15 real-time LMP, hourly mean"),
    TaskSpec("uk_carbon_1h", "hours", "public", 3600.0, (H_,), (), (), "GB carbon intensity, hourly mean"),
]}
SIM_TASK_KIND = {"sim_rack_power_1s": "rack_power_1s", "sim_rack_temp_1min": "rack_temp_1min", "sim_site_power_1h": "site_power_1h"}


@dataclass
class Recipe:
    """Everything that decides which windows exist. Recorded in every result file."""

    name: str = "full"
    train_seeds: tuple = S.TRAIN_SEEDS
    val_seeds: tuple = S.VAL_SEEDS
    test_seeds: tuple = S.TEST_SEEDS
    #: origins per (seed, unit) for each simulator quantity and geometry
    sim_origins: dict = field(default_factory=lambda: {
        "rack_power_1s": {(256, 300): 8, (2048, 256): 4},
        "rack_temp_1min": {(96, 60): 12, (2880, 240): 6},
        "site_power_1h": {(96, 48): 8, (72, 24): 12},
    })
    use_mit: bool = True
    tasks: tuple = tuple(TASKS)
    pool_cap: int = 2000          # a series counts for at most this many origins when pools are sampled
    n_val_per_batch: int = 32     # windows per (task, geometry, covariate mode) validation batch
    sim_workers: int = 1

    def signature(self) -> str:
        d = {"sim_origins": {k: {f"{a}x{b}": n for (a, b), n in v.items()} for k, v in self.sim_origins.items()}}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

    def to_json(self) -> dict:
        return {
            "name": self.name, "train_seeds": list(self.train_seeds), "val_seeds": list(self.val_seeds),
            "test_seeds": list(self.test_seeds), "use_mit": self.use_mit, "tasks": list(self.tasks),
            "sim_origins": {k: {f"{a}x{b}": n for (a, b), n in v.items()} for k, v in self.sim_origins.items()},
            "pool_cap": self.pool_cap, "n_val_per_batch": self.n_val_per_batch,
        }


PILOT = Recipe(name="cpu_pilot", train_seeds=tuple(range(0, 8)), val_seeds=(50, 51, 52), test_seeds=S.TEST_SEEDS,
               sim_origins={
                   "rack_power_1s": {(256, 300): 8},
                   "rack_temp_1min": {(96, 60): 12},
                   "site_power_1h": {(96, 48): 8, (72, 24): 12},
               },
               tasks=tuple(t for t in TASKS))


# ------------------------------------------------------------------------------------------------ batches
@dataclass
class Batch:
    task: str
    geometry: tuple
    groups: tuple
    context: np.ndarray            # (B, T) float32
    future: np.ndarray             # (B, H) float32
    known: Optional[np.ndarray]    # (B, C, T + H) float32 or None
    n_const_rows_zeroed: int = 0

    @property
    def size(self) -> int:
        return self.context.shape[0]


def guard_constant_rows(cov: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero every covariate row (window, channel) that is exactly constant and not a 0/1 indicator (NaN stays NaN).

    Timestral normalises known-future rows with the whole-row std clamped at 1e-6, so a constant non-binary row turns
    float32 rounding noise into large inputs; a row of zeros takes the model's indicator path (loc 0, scale 1)."""
    out = np.array(cov, copy=True)
    with np.errstate(all="ignore"):
        mn, mx = np.nanmin(out, axis=-1), np.nanmax(out, axis=-1)
    const = (mn == mx) & ~((mn == 0.0) | (mn == 1.0))
    # a row that is entirely NaN gives mn == mx == NaN, which compares False: left alone
    if const.any():
        for i, j in zip(*np.nonzero(const)):
            r = out[i, j]
            r[~np.isnan(r)] = 0.0
    return out, int(const.sum())


def choose_groups(rng: np.random.Generator, available: Sequence[str], p_none: float = 0.25, p_all: float = 0.25) -> tuple:
    """Covariate subset of one training batch: none (p_none), all (p_all), else a uniformly drawn non-empty subset."""
    avail = tuple(g for g in ALL_GROUPS if g in available)
    if not avail:
        return ()
    u = rng.random()
    if u < p_none:
        return ()
    if u < p_none + p_all or len(avail) == 1:
        return avail
    while True:
        pick = tuple(g for g in avail if rng.random() < 0.5)
        if pick:
            return pick


def _assemble(task: str, geom: tuple, ctx: np.ndarray, fut: np.ndarray, cov: dict, groups: tuple) -> Batch:
    known, nz = None, 0
    if groups:
        known = np.concatenate([cov[g] for g in groups], axis=1).astype(np.float32)
        known, nz = guard_constant_rows(known)
    return Batch(task, geom, tuple(groups), ctx.astype(np.float32), fut.astype(np.float32), known, nz)


# ------------------------------------------------------------------------------------------------ sources
class SimBank:
    """Precomputed simulator windows (SIMULATED plant, public environment) with covariates for one task."""

    def __init__(self, task: TaskSpec, banks: dict) -> None:
        self.task = task
        self.banks = banks    # geom -> dict(ctx, fut, cov{g: (N,C,T+H)}, seed, unit)

    def n_windows(self, gi: int) -> int:
        b = self.banks.get(self.task.geometries[gi])
        return 0 if b is None else b["ctx"].shape[0]

    def sample(self, rng: np.random.Generator, gi: int, n: int, groups: tuple) -> Batch:
        geom = self.task.geometries[gi]
        b = self.banks[geom]
        idx = rng.integers(0, b["ctx"].shape[0], size=n)
        return _assemble(self.task.name, geom, b["ctx"][idx], b["fut"][idx], {g: b["cov"][g][idx] for g in groups}, groups)


@dataclass
class SeriesItem:
    id: str
    group: str
    values: np.ndarray                 # (L,) float32, NaN allowed
    o_lo: int = 0                      # first admissible origin (index of the first forecast step)
    end_hi: Optional[int] = None       # the forecast part must end at or before this index (exclusive)
    cov: Optional[np.ndarray] = None   # (C, L) float32 known-future covariate rows (group L)


class SeriesPool:
    """Windows cut on the fly from a list of series (public telemetry, measured laptop run, grid series)."""

    def __init__(self, task: TaskSpec, items: list, cap: int = 2000) -> None:
        self.task = task
        self.items = items
        self.cap = cap
        self._cache: dict = {}

    def _ranges(self, gi: int):
        if gi in self._cache:
            return self._cache[gi]
        T, H = self.task.geometries[gi]
        lo, hi, w = [], [], []
        for it in self.items:
            end = len(it.values) if it.end_hi is None else min(len(it.values), it.end_hi)
            a, b = max(T, it.o_lo), end - H
            n = max(0, b - a + 1)
            lo.append(a)
            hi.append(b)
            w.append(min(n, self.cap))
        w = np.asarray(w, dtype=float)
        self._cache[gi] = (np.asarray(lo), np.asarray(hi), w)
        return self._cache[gi]

    def n_windows(self, gi: int) -> int:
        lo, hi, _ = self._ranges(gi)
        return int(np.maximum(0, hi - lo + 1).sum())

    def sample(self, rng: np.random.Generator, gi: int, n: int, groups: tuple) -> Batch:
        T, H = self.task.geometries[gi]
        lo, hi, w = self._ranges(gi)
        if w.sum() == 0:
            raise ValueError(f"{self.task.name}: no admissible window for geometry {(T, H)}")
        p = w / w.sum()
        ctxs, futs, covs = [], [], []
        tries = 0
        while len(ctxs) < n:
            tries += 1
            if tries > 200 * n:
                raise RuntimeError(f"{self.task.name}: cannot find {n} valid windows")
            i = int(rng.choice(len(self.items), p=p))
            o = int(rng.integers(lo[i], hi[i] + 1))
            it = self.items[i]
            c, f = it.values[o - T:o], it.values[o:o + H]
            if not np.all(np.isfinite(f)) or np.mean(~np.isfinite(c)) > MAX_CONTEXT_NAN_FRAC:
                continue
            ctxs.append(c)
            futs.append(f)
            if groups:
                covs.append(it.cov[:, o - T:o + H])
        cov = {}
        if groups:
            cov = {"L": np.stack(covs).astype(np.float32)} if "L" in groups else {}
        return _assemble(self.task.name, (T, H), np.stack(ctxs), np.stack(futs), cov, groups)


# ------------------------------------------------------------------------------------------------ public / measured / grid pools
def _datasets():
    from heatwave.forecast import datasets as D

    return D


def _val_node_sessions() -> tuple[list, list]:
    D = _datasets()
    tr = D.node_sessions("train")
    val = [s for i, s in enumerate(tr) if i % 4 == 1]
    return [s for s in tr if s not in val], val


def _nlr_split() -> tuple[list, list, list]:
    """(train, val, test) NLR run names: test = project rule; val = the repeat just below the test repeat of each group."""
    from heatwave.data import public as P

    D = _datasets()
    r = P.list_nlr_runs()
    r = r[r.kind == "training"].copy()
    test = set(D.nlr_run_names("test"))
    val = []
    for _, g in r.groupby(["model", "nodes"]):
        reps = sorted(g.repeat.to_numpy())
        trep = 4.0 if 4.0 in reps else reps[-1]
        below = [x for x in reps if x < trep]
        if below:
            val += g[g.repeat == below[-1]].run.tolist()
    val = sorted(set(val) - test)
    train = sorted(set(r.run) - test - set(val))
    return train, val, sorted(test)


def _mit_split() -> tuple[list, list, list]:
    D = _datasets()
    ids_tr = D.mit_job_ids("train")
    val = [j for j in ids_tr if j % 4 == 1]
    train = [j for j in ids_tr if j % 4 != 1]
    return train, val, D.mit_job_ids("test")


def build_pools(split: str, recipe: Recipe) -> dict:
    """SeriesPools of the non-simulator tasks for split 'train' or 'val'. TEST parts are never read here."""
    assert split in ("train", "val")
    D = _datasets()
    pools = {}
    want = set(recipe.tasks)

    if "elsayed_gpu_power_1s" in want:
        tr, va = _val_node_sessions()
        keep = set(tr if split == "train" else va)
        items = [SeriesItem(s.id, s.group, s.values.astype(np.float32)) for s in D.elsayed_gpu_series("train", "power") if s.group in keep]
        pools["elsayed_gpu_power_1s"] = SeriesPool(TASKS["elsayed_gpu_power_1s"], items, recipe.pool_cap)

    if "nlr_job_power_1s" in want:
        tr, va, _ = _nlr_split()
        keep = set(tr if split == "train" else va)
        items = [SeriesItem(s.id, s.group, s.values.astype(np.float32)) for s in D.nlr_series("train") if s.group in keep]
        pools["nlr_job_power_1s"] = SeriesPool(TASKS["nlr_job_power_1s"], items, recipe.pool_cap)

    if recipe.use_mit and ("mit_node_power_1s" in want or "mit_node_temp_1min" in want):
        tr, va, _ = _mit_split()
        keep = set(str(j) for j in (tr if split == "train" else va))
        if "mit_node_power_1s" in want:
            items = [SeriesItem(s.id, s.group, s.values.astype(np.float32)) for s in D.mit_series("train", "power") if s.group in keep]
            pools["mit_node_power_1s"] = SeriesPool(TASKS["mit_node_power_1s"], items, recipe.pool_cap)
        if "mit_node_temp_1min" in want:
            items = [SeriesItem(s.id, s.group, s.values.astype(np.float32)) for s in D.mit_series("train", "temp") if s.group in keep]
            pools["mit_node_temp_1min"] = SeriesPool(TASKS["mit_node_temp_1min"], items, recipe.pool_cap)

    if "laptop_power_1s" in want:
        s = D.laptop_series("run1", "power", True)[0]
        n = len(s.values)
        cut = int(0.75 * n)
        cov = s.cov[:1].astype(np.float32)          # training_on only (scripted schedule); ckpt_scheduled is not used
        if split == "train":
            it = SeriesItem("laptop_run1", "laptop_run1", s.values.astype(np.float32), 0, cut, cov)
        else:
            it = SeriesItem("laptop_run1", "laptop_run1", s.values.astype(np.float32), cut, None, cov)
        pools["laptop_power_1s"] = SeriesPool(TASKS["laptop_power_1s"], [it], recipe.pool_cap)

    for name, builder in (("caiso_rt_price_1h", D.caiso_price_series), ("uk_carbon_1h", D.uk_carbon_series)):
        if name not in want:
            continue
        s = builder(False)[0]
        t0 = s.t0
        i_val = int((S.SPLIT_END_VAL - t0) / pd.Timedelta("1h"))      # 2026-04-01
        i_test = int((S.TEST_FROM_UTC - t0) / pd.Timedelta("1h"))     # 2026-05-01
        if split == "train":
            it = SeriesItem(s.id, s.group, s.values.astype(np.float32), 0, i_val)
        else:
            it = SeriesItem(s.id, s.group, s.values.astype(np.float32), i_val, i_test)
        pools[name] = SeriesPool(TASKS[name], [it], recipe.pool_cap)
    return pools


# ------------------------------------------------------------------------------------------------ simulator banks
def _cut_seed(seed: int, recipe: Recipe, cache_dir: str, fx: float, scenario: dict) -> dict:
    """Windows of one simulator seed for every sim task and geometry. Origins are drawn once per (seed, task, geometry)
    and shared by all units; planning cost is shared across origins with the same number of known jobs."""
    S.run_sim(seed, cache_dir, fx)
    d = S.load_sim(seed, cache_dir)
    cov = S.SimCovariates(seed, scenario, fx)
    out: dict = {}
    for ti, (tname, kind) in enumerate(SIM_TASK_KIND.items()):
        if tname not in recipe.tasks:
            continue
        spec = S.SIM_KINDS[kind]
        series = d[spec["key"]].astype(np.float32)          # (L, units)
        L, U = series.shape
        for (T, H), n_orig in recipe.sim_origins[kind].items():
            rng = np.random.default_rng([seed, ti, T, H])
            adm = np.arange(T, L - H + 1)
            origs = np.sort(rng.choice(adm, size=min(n_orig, len(adm)), replace=False))
            ctx, fut, units = [], [], []
            covs = {g: [] for g in S.GROUPS}
            for o in origs:
                c = cov.window(spec["level"], spec["step_s"], int(o), T, H)
                for u in range(U):
                    f = series[o:o + H, u]
                    if not np.all(np.isfinite(f)):
                        continue
                    ctx.append(series[o - T:o, u])
                    fut.append(f)
                    units.append(u)
                    for g in S.GROUPS:
                        covs[g].append(c[g][:, u, :])
            out[f"{tname}|{T}x{H}"] = {
                "ctx": np.stack(ctx).astype(np.float32), "fut": np.stack(fut).astype(np.float32),
                "unit": np.asarray(units, dtype=np.int16),
                "cov": {g: np.stack(v).astype(np.float32) for g, v in covs.items()},
            }
    return out


def _bank_file(cache_dir: str | Path, seed: int, recipe: Recipe) -> Path:
    return Path(cache_dir) / f"ft_bank_{recipe.signature()}_{seed}.npz"


def _save_bank(path: Path, bank: dict) -> None:
    flat = {}
    for k, v in bank.items():
        flat[f"{k}|ctx"], flat[f"{k}|fut"], flat[f"{k}|unit"] = v["ctx"], v["fut"], v["unit"]
        for g, a in v["cov"].items():
            flat[f"{k}|cov_{g}"] = a
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **flat)
    os.replace(tmp, path)


def _load_bank(path: Path) -> dict:
    z = np.load(path)
    out: dict = {}
    for key in z.files:
        k, field_ = key.rsplit("|", 1)
        b = out.setdefault(k, {"cov": {}})
        if field_.startswith("cov_"):
            b["cov"][field_[4:]] = z[key]
        else:
            b[field_] = z[key]
    return out


def _seed_job(args):
    seed, recipe, cache_dir, fx, scenario = args
    p = _bank_file(cache_dir, seed, recipe)
    if not p.exists():
        _save_bank(p, _cut_seed(seed, recipe, cache_dir, fx, scenario))
    return seed, str(p)


def build_sim_banks(seeds: Sequence[int], recipe: Recipe, cache_dir: str | Path, log=print) -> dict:
    """{task: SimBank} for the given simulator seeds (runs the simulator for seeds not yet cached)."""
    scenario = S.load_scenario()
    fx = S.usd_per_eur()[0]
    jobs = [(int(s), recipe, str(cache_dir), fx, scenario) for s in seeds]
    done = {}
    if recipe.sim_workers > 1 and len(jobs) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(recipe.sim_workers) as pool:
            for seed, p in pool.imap_unordered(_seed_job, jobs):
                done[seed] = p
                log(f"sim bank seed {seed} ready ({len(done)}/{len(jobs)})")
    else:
        for j in jobs:
            seed, p = _seed_job(j)
            done[seed] = p
            log(f"sim bank seed {seed} ready ({len(done)}/{len(jobs)})")
    merged: dict = {}
    for seed in seeds:
        bank = _load_bank(Path(done[int(seed)]))
        for k, v in bank.items():
            m = merged.setdefault(k, {"ctx": [], "fut": [], "unit": [], "seed": [], "cov": {g: [] for g in S.GROUPS}})
            m["ctx"].append(v["ctx"]); m["fut"].append(v["fut"]); m["unit"].append(v["unit"])
            m["seed"].append(np.full(len(v["unit"]), seed, dtype=np.int16))
            for g in S.GROUPS:
                m["cov"][g].append(v["cov"][g])
    banks: dict = {}
    for tname in SIM_TASK_KIND:
        per_geom = {}
        for geom in TASKS[tname].geometries:
            k = f"{tname}|{geom[0]}x{geom[1]}"
            if k in merged:
                m = merged[k]
                per_geom[geom] = {"ctx": np.concatenate(m["ctx"]), "fut": np.concatenate(m["fut"]),
                                  "unit": np.concatenate(m["unit"]), "seed": np.concatenate(m["seed"]),
                                  "cov": {g: np.concatenate(a) for g, a in m["cov"].items()}}
        if per_geom:
            banks[tname] = SimBank(TASKS[tname], per_geom)
    return banks


# ------------------------------------------------------------------------------------------------ mixture / validation
class Mixture:
    """Draws training batches: a task by weight, a geometry by the task's probabilities, a covariate subset."""

    def __init__(self, sources: dict, weights: Optional[dict] = None, batch_size: int = 32, p_none: float = 0.25, p_all: float = 0.25,
                 max_context: Optional[int] = None):
        self.max_context = max_context
        self.sources = {k: v for k, v in sources.items()
                        if any(v.n_windows(i) > 0 and (max_context is None or v.task.geometries[i][0] <= max_context)
                               for i in range(len(v.task.geometries)))}
        self.names = sorted(self.sources)
        w = np.asarray([(weights or {}).get(n, 1.0) for n in self.names], dtype=float)
        self.p = w / w.sum()
        self.batch_size = batch_size
        self.p_none, self.p_all = p_none, p_all

    def next(self, rng: np.random.Generator, batch_size: Optional[int] = None) -> Batch:
        name = self.names[int(rng.choice(len(self.names), p=self.p))]
        src = self.sources[name]
        probs = src.task.probs().copy()
        for i in range(len(probs)):
            if src.n_windows(i) == 0 or (self.max_context is not None and src.task.geometries[i][0] > self.max_context):
                probs[i] = 0.0
        gi = int(rng.choice(len(probs), p=probs / probs.sum()))
        groups = choose_groups(rng, src.task.groups, self.p_none, self.p_all)
        return src.sample(rng, gi, batch_size or self.batch_size, groups)

    def describe(self) -> dict:
        return {n: {"weight": float(p), "windows_per_geometry": {f"{g[0]}x{g[1]}": int(self.sources[n].n_windows(i))
                                                                  for i, g in enumerate(self.sources[n].task.geometries)}}
                for n, p in zip(self.names, self.p)}


def validation_batches(sources: dict, seed: int, n: int) -> list:
    """Fixed validation batches: for every task its first (short) geometry, covariates none and (if any) all groups."""
    out = []
    for ti, name in enumerate(sorted(sources)):
        src = sources[name]
        gi = next((i for i in range(len(src.task.geometries)) if src.n_windows(i) > 0), None)
        if gi is None:
            continue
        for mi, groups in enumerate(((), tuple(g for g in ALL_GROUPS if g in src.task.groups))):
            if mi == 1 and not groups:
                continue
            rng = np.random.default_rng([seed, ti, mi])
            out.append(src.sample(rng, gi, min(n, max(1, src.n_windows(gi))), groups))
    return out
