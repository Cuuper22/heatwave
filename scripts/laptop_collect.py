"""Collect a MEASURED laptop GPU power/thermal trace under a scripted training load.

Standalone (needs only torch with CUDA, numpy, nvidia-ml-py). Run on the
laptop:  python laptop_collect.py --out <dir> [--scale 1.0]

The load follows a fixed segment schedule chosen to make thermal dynamics
identifiable for RC calibration: idle baselines, long steady training with
periodic checkpoints (the e002 power shape), and a 60 s on / 60 s off square
wave. Telemetry is sampled at 10 Hz through NVML in a background thread.

Outputs in --out:
  telemetry.csv  utc, mono_s, power_w, energy_mj, temp_c, sm_clock_mhz,
                 util_gpu_pct, util_mem_pct, enforced_limit_w, clock_event_reasons
  events.csv     utc, mono_s, segment, kind (segment_start/step/ckpt_start/ckpt_end), step
  meta.json      GPU name, driver, limits, schedule, torch version
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone

import pynvml
import torch
import torch.nn as nn

# (name, seconds, mode, batch, ckpt_every_s)
SCHEDULE = [
    ("idle-0", 180, "idle", 0, 0),
    ("steady-big-ckpt60", 600, "train", 64, 60),
    ("idle-1", 180, "idle", 0, 0),
    ("steady-mid-ckpt20", 480, "train", 24, 20),
    ("idle-2", 240, "idle", 0, 0),
    ("square-60on-60off", 480, "square", 64, 0),
    ("idle-3", 300, "idle", 0, 0),
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Sampler(threading.Thread):
    def __init__(self, path: str, hz: float = 10.0):
        super().__init__(daemon=True)
        self.path, self.dt, self.stop_flag = path, 1.0 / hz, threading.Event()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _safe(self, fn, *a):
        try:
            return fn(*a)
        except pynvml.NVMLError:
            return ""

    def run(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utc", "mono_s", "power_w", "energy_mj", "temp_c", "sm_clock_mhz",
                        "util_gpu_pct", "util_mem_pct", "enforced_limit_w", "clock_event_reasons"])
            nxt, last_flush = time.monotonic(), time.monotonic()
            while not self.stop_flag.is_set():
                h = self.h
                p = self._safe(pynvml.nvmlDeviceGetPowerUsage, h)
                e = self._safe(pynvml.nvmlDeviceGetTotalEnergyConsumption, h)
                t = self._safe(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU)
                c = self._safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_SM)
                u = self._safe(pynvml.nvmlDeviceGetUtilizationRates, h)
                lim = self._safe(pynvml.nvmlDeviceGetEnforcedPowerLimit, h)
                rsn = self._safe(pynvml.nvmlDeviceGetCurrentClocksEventReasons, h) if hasattr(
                    pynvml, "nvmlDeviceGetCurrentClocksEventReasons") else ""
                w.writerow([utc(), f"{time.monotonic():.3f}",
                            p / 1000 if p != "" else "", e, t, c,
                            u.gpu if u != "" else "", u.memory if u != "" else "",
                            lim / 1000 if lim != "" else "", rsn])
                if time.monotonic() - last_flush > 1:
                    f.flush()
                    last_flush = time.monotonic()
                nxt += self.dt
                time.sleep(max(0.0, nxt - time.monotonic()))


class TinyLM(nn.Module):
    def __init__(self, vocab=8192, d=512, layers=6, heads=8, ctx=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x) + self.pos[: x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        return self.head(self.body(h, mask=mask, is_causal=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply segment durations")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    dev = torch.device("cuda")
    model = TinyLM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    meta = {
        "gpu": pynvml.nvmlDeviceGetName(h),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "limits_w": [x / 1000 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)],
        "default_limit_w": pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000,
        "torch": torch.__version__,
        "params": sum(p.numel() for p in model.parameters()),
        "schedule": SCHEDULE,
        "scale": args.scale,
        "started_utc": utc(),
        "data_kind": "measured",
    }
    sampler = Sampler(os.path.join(args.out, "telemetry.csv"))
    sampler.start()
    ckdir = tempfile.mkdtemp(prefix="hw_ckpt_")
    step = 0
    with open(os.path.join(args.out, "events.csv"), "w", newline="") as ef:
        ew = csv.writer(ef)
        ew.writerow(["utc", "mono_s", "segment", "kind", "step"])

        def ev(seg, kind):
            ew.writerow([utc(), f"{time.monotonic():.3f}", seg, kind, step])

        def train_step(batch):
            nonlocal step
            x = torch.randint(0, 8192, (batch, 256), device=dev)
            loss = nn.functional.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            step += 1
            if step % 10 == 0:
                ev(seg, "step")

        def checkpoint():
            ev(seg, "ckpt_start")
            torch.save({"m": model.state_dict(), "o": opt.state_dict()}, os.path.join(ckdir, "ck.pt"))
            ev(seg, "ckpt_end")

        try:
            for seg, secs, mode, batch, ck in SCHEDULE:
                secs = secs * args.scale
                ev(seg, "segment_start")
                ef.flush()
                t0 = time.monotonic()
                last_ck = t0
                while time.monotonic() - t0 < secs:
                    el = time.monotonic() - t0
                    if mode == "idle" or (mode == "square" and int(el // 60) % 2 == 1):
                        time.sleep(0.05)
                        continue
                    train_step(batch)
                    if ck and time.monotonic() - last_ck >= ck:
                        checkpoint()
                        last_ck = time.monotonic()
                ef.flush()
        finally:
            ev("end", "segment_start")
            sampler.stop_flag.set()
            sampler.join(timeout=5)
            meta["ended_utc"] = utc()
            meta["steps"] = step
            with open(os.path.join(args.out, "meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
