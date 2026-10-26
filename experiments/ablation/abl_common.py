"""Helpers shared by the ablation scripts."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

#: Forward-addition order, declared before any score was computed. Plan-derived covariates (job schedule, then the
#: training-step phase) come first because they describe the load directly; the weather forecast follows (it drives
#: cooling power and temperature); price/carbon come last (the simulated controller does not react to price or carbon,
#: so they have no causal path to power or temperature).
FORWARD_ORDER = ("J", "P", "W", "C")


def sim_configs(order=FORWARD_ORDER):
    """Unique covariate sets of the ablation and the roles each one plays.

    Returns (configs, roles): configs is a list of (name, groups) in a fixed order, roles maps name -> list of role
    strings ('alone', 'forward_k', 'all', 'leave_out_X').
    """
    sets, roles = {"none": ()}, {"none": ["baseline"]}

    def add(groups, role):
        key = "+".join(groups) if groups else "none"
        sets.setdefault(key, tuple(groups))
        roles.setdefault(key, []).append(role)

    for g in order:
        add((g,), "alone")
    for k in range(1, len(order) + 1):
        add(tuple(order[:k]), "forward_%d" % k if k < len(order) else "forward_%d_all" % k)
    add(tuple(order), "all")
    for g in order:
        add(tuple(x for x in order if x != g), "leave_out_" + g)
    names = list(sets)
    # stable display order: none, alone, forward, all, leave-out
    return [(n, sets[n]) for n in names], roles


def hardware_info(device: str) -> dict:
    info = {"device": device, "python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        if device.startswith("cuda") and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def make_timestral(size: str, weights_dir: str | None, device: str, threads: int):
    from heatwave.forecast.timestral_fc import TimestralForecaster

    return TimestralForecaster(size, weights_dir=weights_dir, device=device, threads=threads)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
