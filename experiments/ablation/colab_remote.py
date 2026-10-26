"""Runs ON the Colab VM (sent with `colab exec -f`). Steps are chosen by the STEP variable set by the caller.

STEP=setup     : unpack the code tarballs under /content and install the packages
STEP=simcache  : run sim_cache.py for the 20 (variant, seed) pairs in parallel worker processes
STEP=smoke     : tiny GPU run of both ablation scripts (checks cuda and measures throughput)
STEP=full      : start the full ablation_sim.py and ablation_real.py runs in the background (logs under /content/out)
STEP=check     : print cores, memory, GPU and import check
STEP=status    : print log tails
"""

import os
import subprocess
import sys

STEP = os.environ.get("STEP", "status")
WT = "/content/heatwave-ablation"
GS = "/content/gpu_stack"
OUT = "/content/out"
ENV = {**os.environ, "PYTHONPATH": WT, "HEATWAVE_SIM_SCENARIO": f"{GS}/experiments/e003-heatwave/sim-scenario-v2.json"}


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, env=ENV, capture_output=True, text=True, **kw)
    print(r.stdout, r.stderr, flush=True)
    return r


os.makedirs(OUT, exist_ok=True)
if STEP == "setup":
    for d, t in ((WT, "heatwave-ablation.tar.gz"), (GS, "gpu_stack.tar.gz")):
        os.makedirs(d, exist_ok=True)
        sh(f"tar xzf /content/{t} -C {d}")
    sh("pip install -q git+https://github.com/mistralai/timestral.git numpy pandas pyarrow scipy statsforecast sympy 2>&1 | tail -3")
    sh(f"pip install -q --no-deps -e {WT} -e {GS} 2>&1 | tail -3")
    sh("nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv")
elif STEP == "simcache":
    workers = int(os.environ.get("WORKERS", "4"))
    jobs = [(v, s) for v in ("literal", "worlds") for s in range(100, 110)]
    procs = []
    for k in range(workers):
        mine = jobs[k::workers]
        by_v = {}
        for v, s in mine:
            by_v.setdefault(v, []).append(s)
        cmd = " && ".join(
            f"python {WT}/experiments/ablation/sim_cache.py --variant {v} --seeds {' '.join(map(str, ss))} --out {WT}/results/ablation/cache"
            for v, ss in by_v.items())
        procs.append(subprocess.Popen(f"nohup sh -c '{cmd}' > {OUT}/simcache_{k}.log 2>&1", shell=True, env=ENV))
    print("started", len(procs), "workers")
elif STEP == "smoke":
    which = os.environ.get("SMOKE", "sim")
    if which == "sim":
        cmd = (f"python experiments/ablation/ablation_sim.py --variants worlds --kinds {os.environ.get('KIND', 'rack_power_1s')} --seeds 100 --sizes 209M "
               f"--device cuda --stride-mult 30 --reps 20 --out {OUT}/smoke_sim.json")
    else:
        cmd = (f"python experiments/ablation/ablation_real.py --sizes 209M --device cuda --stride-mult 40 --reps 20 --out {OUT}/smoke_real.json")
    subprocess.Popen(f"cd {WT} && nohup {cmd} > {OUT}/smoke_{which}.log 2>&1", shell=True, env=ENV)
    print("started", cmd)
elif STEP == "full":
    extra = os.environ.get("EXTRA", "")
    sh(f"cd {WT} && nohup python experiments/ablation/ablation_sim.py --sizes 209M 28M --device cuda --batch-size {os.environ.get('BS', '64')} "
       f"--run-label colab-a100 --out {WT}/results/ablation/ablation_sim.json {extra} > {OUT}/full_sim.log 2>&1 &")
    sh(f"cd {WT} && nohup python experiments/ablation/ablation_real.py --sizes 209M 28M --device cuda --batch-size {os.environ.get('BS', '64')} "
       f"--run-label colab-a100 --out {WT}/results/ablation/ablation_real.json > {OUT}/full_real.log 2>&1 &")
elif STEP == "check":
    sh("nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh(f"cd {WT} && python -c 'import timestral, heatwave, gpu_stack; from heatwave.forecast import covariates; print(1)'")
else:
    sh(f"tail -n 4 {OUT}/*.log; ls -la {WT}/results/ablation {WT}/results/ablation/cache | head -60; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv")
