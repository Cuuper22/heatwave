"""Write notebooks/finetune_timestral_colab.ipynb (the Colab copy of the fine-tune recipe).

The notebook runs the same script as the CPU pilot (experiments/finetune/run_recipe.py) on a Colab GPU runtime. All code
cells are plain Python (shell steps go through ``sh(...)``), so each cell can be syntax-checked with ``ast.parse``.
Regenerate with: python experiments/finetune/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "finetune_timestral_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Fine-tune Timestral 28M, 123M and 209M on the Heatwave data recipe (Colab GPU runtime)

What this notebook does: builds the training windows (simulated cluster on public weather / price / carbon, public GPU
telemetry, measured laptop run, grid series), fine-tunes Timestral 28M, 123M and 209M with the pinball loss on the
horizon, and scores zero-shot against fine-tuned on the held-out windows (the same TEST windows for both). It writes a
results JSON in the schema of `results/finetune/cpu_pilot.json` and the fine-tuned weights in the Hugging Face layout
(`config.json` + `model.safetensors`), which `heatwave.forecast.timestral_fc.TimestralForecaster(size, weights_dir=...)` loads
unchanged.

Runtime: GPU (A100 recommended; the 209M run needs about 20 GB). Everything it needs, from you:

1. **heatwave code bundle**: `git -C <heatwave worktree> archive --format=tar.gz -o heatwave_bundle.tar.gz HEAD` (includes the
   committed public data under `data/processed`), uploaded to `/content/heatwave_bundle.tar.gz`; or set `HEATWAVE_GIT_URL`
   in the next cell (and leave the upload out).
2. **gpu_stack branch `heatwave`** (the simulator): `git -C <gpu_stack> archive --format=tar.gz -o gpu_stack_bundle.tar.gz heatwave`,
   uploaded to `/content/gpu_stack_bundle.tar.gz`; or set `GPU_STACK_GIT_URL`.
3. **MIT Supercloud series (optional)**: `tar -czf mit_data.tar.gz -C <heatwave>/data/processed public_gpu_mit public_gpu_mit_jobs.parquet public_gpu_mit_jobs.table.json`,
   uploaded to `/content/mit_data.tar.gz`. The licence is CC BY-NC-ND 4.0: keep the files private, never commit them. Without
   this upload the recipe skips the two MIT tasks and the MIT evaluation sets (recorded in the results JSON).

Licence of the model: Timestral is released under a Modified MIT License (Mistral AI): the upstream LICENSE is copied next to every
fine-tuned checkpoint, and the licence's revenue restriction applies to derivatives (see the Hub model card).
"""),
    code("""
# ---- settings (edit here) ----
SIZES = ["28M", "123M", "209M"]          # one full fine-tune per size
RECIPE = "full"                          # "full": simulator seeds 0-49 train, 50-59 validation; "pilot": a small subset for a smoke test
STEPS = 3000                             # maximum optimiser steps per size (early stopping on the validation set applies)
BATCH_SIZE = 64
EVAL_EVERY = 100
PATIENCE = 6
SWEEP_STEPS = 300                        # learning-rate sweep (validation set only), per size
SIM_WORKERS = 8                          # processes for the simulator runs (the A100 runtime has about 12 vCPUs)
HEATWAVE_GIT_URL = ""                    # e.g. "https://github.com/<you>/heatwave.git" (leave empty to use the uploaded bundle)
HEATWAVE_GIT_REF = "w/finetune"
GPU_STACK_GIT_URL = ""                   # e.g. "https://github.com/<you>/gpu_stack-.git" (branch heatwave); empty: uploaded bundle
WORK = "/content/work"
"""),
    code("""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sh(cmd, check=True, cwd=None, env=None):
    \"\"\"Run a shell command, streaming its output.\"\"\"
    e = dict(os.environ, **(env or {}))
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p.returncode


import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.cuda.is_available(), "switch the runtime to a GPU (Runtime > Change runtime type)"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
"""),
    code("""
# ---- install ----
sh(f"{sys.executable} -m pip install -q 'git+https://github.com/mistralai/timestral.git' numpy pandas pyarrow scipy statsforecast sympy statsmodels safetensors huggingface_hub")
"""),
    code("""
# ---- code: heatwave and the simulator (uploaded bundles, or git) ----
ROOT = Path("/content/heatwave")
GS = Path("/content/gpu_stack")
for d, bundle, url, ref in ((ROOT, "/content/heatwave_bundle.tar.gz", HEATWAVE_GIT_URL, HEATWAVE_GIT_REF),
                            (GS, "/content/gpu_stack_bundle.tar.gz", GPU_STACK_GIT_URL, "heatwave")):
    sh(f"rm -rf {d} && mkdir -p {d}")
    if os.path.exists(bundle):
        sh(f"tar -xzf {bundle} -C {d}")
    elif url:
        sh(f"git clone --depth 1 --branch {ref} {url} {d}")
    else:
        raise FileNotFoundError(f"upload {bundle} or set the git URL for {d.name}")
if os.path.exists("/content/mit_data.tar.gz"):
    sh(f"tar -xzf /content/mit_data.tar.gz -C {ROOT}/data/processed")
    USE_MIT = True
else:
    USE_MIT = False
    print("MIT Supercloud series not uploaded: the recipe runs without the MIT tasks")
sh(f"{sys.executable} -m pip install -q --no-deps -e {ROOT} -e {GS}")
commit_file = ROOT / "GIT_COMMIT"
env = {"HEATWAVE_SIM_SCENARIO": str(GS / "experiments/e003-heatwave/sim-scenario-v3.json"),
       "PYTHONPATH": str(ROOT)}
if commit_file.exists():
    env["HEATWAVE_GIT_COMMIT"] = commit_file.read_text().strip()      # git archive bundles carry no .git: record the commit
os.environ.update(env)
print(env)
"""),
    code("""
# ---- sanity: the model loads and a zero-step fine-tune reproduces the zero-shot forecast ----
from huggingface_hub import snapshot_download

snapshot_download("mistralai/Timestral-5M")      # the tests use the 5M checkpoint
sh(f"{sys.executable} -m pytest -x -q tests/test_finetune.py -k 'zero_step or grad_forward or pinball' 2>&1 | tail -5", cwd=ROOT, check=False)
"""),
    code("""
# ---- data: simulator runs on the public environment (seeds 0-49 train, 50-59 validation, 100-109 TEST), window banks ----
nw = os.cpu_count() or 2
sh(f"{sys.executable} experiments/finetune/run_recipe.py --recipe {RECIPE} --stage data --device cuda --sim-workers {min(SIM_WORKERS, nw)} "
   f"{'' if USE_MIT else '--no-mit'} --cache-dir {WORK}/cache --work-dir {WORK}/weights", cwd=ROOT)
"""),
    code("""
# ---- learning-rate sweep (validation only), fine-tune, evaluate zero-shot vs fine-tuned on the TEST windows ----
runs = " ".join(f"{s}-full" for s in SIZES)
out_json = f"{WORK}/results/finetune_colab.json"
sh(f"{sys.executable} experiments/finetune/run_recipe.py --recipe {RECIPE} --stage all --device cuda --runs {runs} "
   f"--steps {STEPS} --batch-size {BATCH_SIZE} --eval-every {EVAL_EVERY} --patience {PATIENCE} --warmup 100 "
   f"--sweep-steps {SWEEP_STEPS} --sim-workers {min(SIM_WORKERS, nw)} {'' if USE_MIT else '--no-mit'} "
   f"--cache-dir {WORK}/cache --work-dir {WORK}/weights --out {out_json} --run-label colab-notebook", cwd=ROOT)
"""),
    code("""
# ---- summary: mean WQL over the datasets (none = no covariates) ----
res = json.loads(Path(out_json).read_text())["result"]
for size, ev in res["evaluation"].items():
    for mode, models in ev["macro"].items():
        print(f"[{size}] mode={mode}")
        for label, m in models.items():
            extra = f"  ratio to zero-shot {m['mean_wql_ratio_to_reference']:.4f}, datasets with lower WQL {m['n_datasets_lower_wql']}/{len(m['datasets'])}" if "mean_wql_ratio_to_reference" in m else ""
            print(f"   {label:28s} mean WQL {m['mean_wql']:.4f}{extra}")
"""),
    code("""
# ---- files to copy back ----
# 1) results JSON (commit to results/finetune/colab/ in the heatwave repo; its provenance records the archive commit)
# 2) the training histories (small; commit next to it)
# 3) the weights directories (large: do not commit; keep them and pass weights_dir=... to TimestralForecaster)
import shutil

hist_dir = Path(WORK) / "results" / "histories"
hist_dir.mkdir(parents=True, exist_ok=True)
for s in SIZES:
    h = Path(WORK) / "weights" / RECIPE / f"{s}-full" / "history.json"
    if h.exists():
        shutil.copy(h, hist_dir / f"{s}-full_history.json")
to_copy = [Path(out_json)] + sorted(hist_dir.glob("*.json"))
print("Copy these into the repo (results/finetune/colab/):")
for f in to_copy:
    print(f"  {f}  ({f.stat().st_size / 1e3:.0f} kB)")
print()
print("Keep these weight directories (Hub layout, upstream LICENSE included); do not commit them:")
for s in SIZES:
    d = Path(WORK) / "weights" / RECIPE / f"{s}-full" / "best"
    if d.exists():
        print(f"  {d}  ({sum(p.stat().st_size for p in d.iterdir()) / 1e6:.0f} MB)")
sh(f"cd {WORK} && tar -cf finetuned_weights.tar " + " ".join(f"weights/{RECIPE}/{s}-full/best" for s in SIZES) + " && ls -la finetuned_weights.tar", check=False)
print("download with the Colab file browser or: from google.colab import files; files.download(...)")
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
