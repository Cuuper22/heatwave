"""Fine-tune Timestral on the Heatwave data recipe and score zero-shot against fine-tuned on the held-out windows.

One script for the CPU pilot (VM, ``--recipe pilot``) and the Colab GPU run (``--recipe full``); the Colab notebook
calls this same script. Stages (``--stage``): ``data`` (simulator runs, window banks), ``sweep`` (learning-rate sweep on the
validation set only), ``train``, ``eval``, or ``all``. Everything is resumable: simulator runs, banks and finished runs are
cached; an interrupted run continues from its ``train_state.pt``.

Data kinds: the simulator windows are SIMULATED plant + PUBLIC environment, the telemetry is PUBLIC, the laptop runs
are MEASURED; the result file is labelled "mixed" and every dataset carries its own kind.
Splits: see heatwave/finetune/data.py (train / validation / TEST; the TEST parts are read only by the ``eval`` stage).

Usage:
  CPU pilot : PYTHONPATH=. nice -n 10 python experiments/finetune/run_recipe.py --recipe pilot --runs 5M-full 5M-lora 28M-head \
              --out results/finetune/cpu_pilot.json
  Colab     : python experiments/finetune/run_recipe.py --recipe full --device cuda --runs 28M-full 123M-full 209M-full \
              --out results/finetune/colab/finetune_colab.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatwave.finetune import data as Dt  # noqa: E402
from heatwave.finetune import evaluate as Ev  # noqa: E402
from heatwave.finetune import sim as S  # noqa: E402
from heatwave.finetune import train as Tr  # noqa: E402
from heatwave.provenance import write_result  # noqa: E402

#: run name -> TrainConfig fields. The learning rate is chosen by the sweep (validation only) unless --lr is given.
RUNS = {
    "5M-full": dict(base="5M", mode="full"),
    "5M-lora": dict(base="5M", mode="lora", lora_r=8, lora_alpha=16.0),
    "28M-head": dict(base="28M", mode="head", last_blocks=2),
    "28M-full": dict(base="28M", mode="full"),
    "28M-lora": dict(base="28M", mode="lora", lora_r=8, lora_alpha=16.0),
    "123M-full": dict(base="123M", mode="full"),
    "209M-full": dict(base="209M", mode="full"),
    "209M-lora": dict(base="209M", mode="lora", lora_r=8, lora_alpha=16.0),
}
DEFAULT_SWEEP_LR = {"full": (1e-5, 3e-5, 1e-4), "head": (1e-4, 3e-4, 1e-3), "lora": (1e-4, 3e-4, 1e-3)}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hardware(device: str) -> dict:
    info = {"device": device, "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "cpu_count": os.cpu_count()}
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    if os.environ.get("COLAB_RELEASE_TAG") or os.path.exists("/content"):
        info["environment"] = "Google Colab"
    return info


def make_recipe(name: str, args) -> Dt.Recipe:
    r = Dt.PILOT if name == "pilot" else Dt.Recipe(name="full")
    r = replace(r, sim_workers=args.sim_workers, use_mit=not args.no_mit)
    if args.n_val:
        r = replace(r, n_val_per_batch=args.n_val)
    return r


def build_train_data(recipe: Dt.Recipe, cache_dir: Path):
    log(f"simulator banks: train seeds {len(recipe.train_seeds)}, validation seeds {len(recipe.val_seeds)}")
    tr_sim = Dt.build_sim_banks(recipe.train_seeds, recipe, cache_dir, log)
    va_sim = Dt.build_sim_banks(recipe.val_seeds, recipe, cache_dir, log)
    tr_src = {**tr_sim, **Dt.build_pools("train", recipe)}
    va_src = {**va_sim, **Dt.build_pools("val", recipe)}
    return tr_src, va_src


def make_cfg(run: str, args, lr: float, steps: int, device: str) -> Tr.TrainConfig:
    spec = dict(RUNS[run])
    return Tr.TrainConfig(lr=lr, max_steps=steps, batch_size=args.batch_size, eval_every=args.eval_every, patience=args.patience,
                          warmup_steps=args.warmup, symmetric=not args.raw_pass, seed=args.seed, device=device,
                          max_seconds=args.max_seconds, checkpoint_every=args.eval_every, log_every=args.log_every,
                          head_lr_mult=args.head_lr_mult, micro_batch=args.micro_batch or None, **spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--runs", nargs="+", default=["5M-full"], choices=list(RUNS))
    ap.add_argument("--stage", choices=["data", "sweep", "train", "eval", "all"], default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sim-workers", type=int, default=1)
    ap.add_argument("--no-mit", action="store_true", help="skip the MIT Supercloud series (licence: local copy only)")
    ap.add_argument("--cache-dir", default=str(ROOT / "results" / "cache" / "finetune"))
    ap.add_argument("--work-dir", default="/home/claude/heatwave-finetune-weights")
    ap.add_argument("--out", default=str(ROOT / "results" / "finetune" / "cpu_pilot.json"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--lr", type=float, default=None, help="skip the sweep and use this learning rate for every run")
    ap.add_argument("--sweep-lrs", type=float, nargs="+", default=None)
    ap.add_argument("--sweep-steps", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=None, help="wall-clock limit per run")
    ap.add_argument("--raw-pass", action="store_true", help="train on the single raw pass instead of the symmetric-averaged forecast")
    ap.add_argument("--head-lr-mult", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=0, help="windows per forward chunk (gradient accumulation); 0: whole batch")
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--run-label", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    recipe = make_recipe(args.recipe, args)
    cache_dir, work = Path(args.cache_dir), Path(args.work_dir) / recipe.name
    work.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    tr_src, va_src = build_train_data(recipe, cache_dir)
    if args.stage == "data":
        log("data stage done")
        return
    mixture = Dt.Mixture(tr_src, batch_size=args.batch_size)
    val_batches = Dt.validation_batches(va_src, seed=1234, n=recipe.n_val_per_batch)
    log(f"mixture: {json.dumps(mixture.describe())}")
    log(f"validation batches: {len(val_batches)}")

    # ---- learning-rate sweep (validation set only)
    chosen: dict = {}
    sweep_out: dict = {}
    for run in args.runs:
        mode = RUNS[run]["mode"]
        if args.lr is not None:
            chosen[run] = args.lr
            continue
        lrs = args.sweep_lrs or DEFAULT_SWEEP_LR[mode]
        sf = work / "sweep" / f"{run}.json"
        if sf.exists():
            sweep_out[run] = json.loads(sf.read_text())
        elif args.stage in ("sweep", "all"):
            res = {}
            for lr in lrs:
                d = work / "sweep" / run / f"lr{lr:g}"
                cfg = make_cfg(run, args, lr, args.sweep_steps, args.device)
                cfg.eval_every = max(10, args.sweep_steps // 2)
                cfg.patience = 100
                cfg.checkpoint_every = 0
                r = Tr.train(cfg, mixture, val_batches, d, log=log, resume=False)
                res[f"{lr:g}"] = {"lr": lr, "best_val_macro": r["best_val_macro"], "initial_val_macro": r["initial_val_macro"],
                                  "best_step": r["best_step"], "seconds": r["seconds"]}
                for f in (d / "best").glob("*.safetensors"):
                    f.unlink()
            sweep_out[run] = res
            sf.parent.mkdir(parents=True, exist_ok=True)
            sf.write_text(json.dumps(res, indent=1) + "\n")
        if run in sweep_out:
            chosen[run] = min(sweep_out[run].values(), key=lambda x: x["best_val_macro"])["lr"]
    if args.stage == "sweep":
        log(f"sweep done: {chosen}")
        return

    # ---- training
    trained: dict = {}
    for run in args.runs:
        d = work / run
        hist = d / "history.json"
        if hist.exists() and json.loads(hist.read_text()).get("stop_reason") in ("max_steps", "early_stopping", "max_seconds") and (d / "best" / "model.safetensors").exists() and not (d / "train_state.pt").exists():
            trained[run] = json.loads(hist.read_text())
            log(f"{run}: finished earlier, reusing")
            continue
        if args.stage not in ("train", "all"):
            continue
        cfg = make_cfg(run, args, chosen[run], args.steps, args.device)
        log(f"=== training {run} lr {cfg.lr:g} steps {cfg.max_steps} ===")
        prov = {"recipe": recipe.name, "git_note": "see result file provenance", "license_note": "derived from Timestral (Mistral AI, Modified MIT)"}
        trained[run] = Tr.train(cfg, mixture, val_batches, d, log=log, resume=True, provenance=prov)
        st = d / "train_state.pt"
        if st.exists():
            st.unlink()                 # finished: the resume state is no longer needed (weights are in best/)
    if args.stage == "train":
        return

    # ---- evaluation on the TEST windows
    log("building held-out evaluation windows")
    eval_sets = Ev.build_eval_sets(recipe, cache_dir, log)
    from heatwave.forecast.timestral_fc import TimestralForecaster as Wrapper

    by_size: dict = {}
    for size in sorted({RUNS[r]["base"] for r in args.runs}, key=lambda s: float(s[:-1])):
        models = {f"zero-shot-{size}": Wrapper(size, device=args.device, threads=args.threads)}
        for run in args.runs:
            if RUNS[run]["base"] == size and run in trained:
                models[f"finetuned-{run}"] = Wrapper(size, weights_dir=str(work / run / "best"), device=args.device, threads=args.threads)
        log(f"evaluating {list(models)}")
        by_size[size] = Ev.evaluate_models(models, eval_sets, n_boot=args.boot, log=log, reference=f"zero-shot-{size}", batch=args.eval_batch)

    weights = {}
    for run in args.runs:
        f = work / run / "best" / "model.safetensors"
        if f.exists():
            weights[run] = {"dir": str(work / run / "best"), "model.safetensors_sha256": sha(f), "bytes": f.stat().st_size,
                            "note": "weights are not committed to git (size); upstream LICENSE is copied next to them"}
    train_summary = {}
    for run, r in trained.items():
        train_summary[run] = {k: r[k] for k in ("config", "trainable", "steps_run", "best_step", "best_val_macro", "initial_val_macro",
                                                "stop_reason", "skipped_batches", "seconds", "device")}
        train_summary[run]["validation_curve"] = [{"step": h["step"], "val_macro": h["val_macro"]} for h in r["history"]]
    result = {
        "what": "Timestral zero-shot versus fine-tuned on the Heatwave recipe, scored on held-out (TEST) windows",
        "recipe": recipe.to_json(),
        "splits": {
            "simulator": "train seeds 0-49 (weeks 2024-02-05 + 14 d x seed), validation 50-59 (2026-01-05 + 7 d x (seed-50)), TEST 100-109 (scenario start_utc + 7 d x (seed-100)); sim-scenario-v3 on the public environment",
            "elsayed": "TEST sessions index%3==0; validation: every 4th remaining session; rest train",
            "nlr": "TEST repeat 4 (or last repeat); validation: the repeat just below; rest train",
            "mit": "TEST job id%4==0; validation id%4==1; rest train (local copy, CC BY-NC-ND 4.0, not committed)",
            "laptop": "run1 train (first 75 %) / validation (rest); run2 TEST",
            "grid": "train: forecast part before 2026-04-01; validation: April 2026; TEST: origins from 2026-05-01",
        },
        "lr_sweep_validation_only": sweep_out, "learning_rates_used": chosen,
        "mixture": mixture.describe(), "n_validation_batches": len(val_batches),
        "training": train_summary, "weights": weights,
        "evaluation": by_size,
        "hardware": hardware(args.device), "run_label": args.run_label,
        "wall_seconds_total": time.time() - t_all,
        "license": "Timestral weights: Modified MIT License (Mistral AI, 2026); derivatives are subject to the same terms, "
                   "including the revenue restriction in clause 2",
    }
    inputs = [ROOT / "data" / "processed" / "laptop_run1_1s.parquet", ROOT / "data" / "processed" / "laptop_run2_1s.parquet"]
    notes = (f"data_kind per dataset (simulated plant on public environment / public / measured); hardware {result['hardware']}; "
             f"recipe {recipe.name}; device {args.device}")
    write_result(args.out, result, script=__file__, data_kind="mixed", inputs=inputs, notes=notes)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
