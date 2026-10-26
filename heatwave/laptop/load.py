"""Training load generator with periodic checkpoints.

A small GPT-style transformer is trained on synthetic random tokens (the
compute pattern is what matters, not the loss) with a synchronous checkpoint
save every N steps. The GPU power trace therefore has the e002 shape: steady
compute with dips while a checkpoint is written.

Every step and checkpoint is timestamped into an events CSV:
    t_utc, t_unix, event, step, value
Events: config (value = checkpoint every N steps), step (value = loss),
ckpt_start, ckpt_end (value = bytes written), end. These are the known-future
"training step phase" and "checkpoint timing" covariates for the forecaster.

Run alone:
    python -m heatwave.laptop.load --minutes 2 --events events.csv --size tiny
"""

from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .state import GuardedThread

EVENT_COLUMNS = ("t_utc", "t_unix", "event", "step", "value")

# name: (d_model, layers, heads, seq_len, batch)
SIZES = {
    "nano": (64, 2, 2, 32, 4),  # CPU smoke tests only
    "tiny": (256, 4, 4, 128, 16),
    "small": (512, 8, 8, 256, 16),
    "medium": (768, 12, 12, 256, 8),
}
VOCAB = 8192


class EventLog:
    """Append-only events CSV, flushed about once per second."""

    def __init__(self, path: str | Path) -> None:
        from .telemetry import iso_utc

        self._iso = iso_utc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._f.write(",".join(EVENT_COLUMNS) + "\n")
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()

    def write(self, event: str, step: int = 0, value: float = 0.0, t_unix: Optional[float] = None) -> None:
        t = time.time() if t_unix is None else t_unix
        with self._lock:
            self._f.write(f"{self._iso(t)},{t:.4f},{event},{step},{value:.6g}\n")
            if time.monotonic() - self._last_flush >= 1.0 or event in ("ckpt_end", "end", "config"):
                self._f.flush()
                self._last_flush = time.monotonic()

    def close(self) -> None:
        with self._lock:
            self._f.flush()
            self._f.close()


@dataclass
class LoadConfig:
    size: str = "small"
    minutes: float = 5.0
    ckpt_every: int = 0           # steps; 0 = calibrate from ckpt_period_s
    ckpt_period_s: float = 30.0   # used when ckpt_every == 0
    device: str = "cuda"
    seed: int = 0
    warmup_steps: int = 10
    stop_file: Optional[str] = None


def build_model(size: str):
    import torch
    from torch import nn

    d, layers, heads, seq, _ = SIZES[size]

    class TinyGPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(VOCAB, d)
            self.pos = nn.Embedding(seq, d)
            layer = nn.TransformerEncoderLayer(
                d, heads, 4 * d, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, VOCAB, bias=False)
            self.head.weight = self.tok.weight
            mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
            self.register_buffer("mask", mask, persistent=False)

        def forward(self, x):
            h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
            h = self.blocks(h, mask=self.mask[: x.shape[1], : x.shape[1]])
            return self.head(self.ln(h))

    return TinyGPT()


def param_count(size: str) -> int:
    m = build_model(size)
    return sum(p.numel() for p in m.parameters())


def _save_checkpoint(model, opt, path: Path) -> int:
    import torch

    with open(path, "wb") as f:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, f)
        f.flush()
        os.fsync(f.fileno())
    return path.stat().st_size


def run_load(cfg: LoadConfig, events_path: str | Path, stop_event: Optional[threading.Event] = None,
             summary_path: Optional[str | Path] = None) -> dict:
    import torch
    import torch.nn.functional as F

    if cfg.size not in SIZES:
        raise ValueError(f"size must be one of {sorted(SIZES)}")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    _, _, _, seq, batch = SIZES[cfg.size]
    model = build_model(cfg.size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    amp = dev.type == "cuda"
    ev = EventLog(events_path)
    stop_event = stop_event or threading.Event()
    t_end = time.monotonic() + cfg.minutes * 60.0

    def should_stop() -> bool:
        return (stop_event.is_set() or time.monotonic() >= t_end
                or (cfg.stop_file is not None and os.path.exists(cfg.stop_file)))

    def train_step() -> float:
        x = torch.randint(0, VOCAB, (batch, seq + 1), device=dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return float(loss.item())

    step_times = []
    step = 0
    ckpt_every = cfg.ckpt_every
    n_ckpt = 0
    ckpt_seconds = []
    with tempfile.TemporaryDirectory(prefix="heatwave_ckpt_") as tmp:
        ckpt_path = Path(tmp) / "ckpt.pt"
        try:
            while not should_stop():
                t0 = time.monotonic()
                loss = train_step()
                step += 1
                if step <= cfg.warmup_steps:
                    step_times.append(time.monotonic() - t0)
                    if step == cfg.warmup_steps:
                        if ckpt_every <= 0:
                            typical = sorted(step_times[len(step_times) // 2:])[len(step_times[len(step_times) // 2:]) // 2]
                            ckpt_every = max(1, int(round(cfg.ckpt_period_s / max(typical, 1e-6))))
                        ev.write("config", step, ckpt_every)
                    continue
                ev.write("step", step, loss)
                if ckpt_every > 0 and step % ckpt_every == 0:
                    ev.write("ckpt_start", step, 0)
                    c0 = time.monotonic()
                    nbytes = _save_checkpoint(model, opt, ckpt_path)
                    ckpt_seconds.append(time.monotonic() - c0)
                    ev.write("ckpt_end", step, nbytes)
                    n_ckpt += 1
        finally:
            ev.write("end", step, 0)
            ev.close()
    out = {
        "size": cfg.size, "device": cfg.device, "params": n_params, "steps": step,
        "ckpt_every_steps": ckpt_every, "checkpoints": n_ckpt,
        "mean_ckpt_seconds": (sum(ckpt_seconds) / len(ckpt_seconds)) if ckpt_seconds else None,
        "peak_cuda_mem_mib": (torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else None,
    }
    if summary_path is not None:
        from heatwave.provenance import write_result

        write_result(summary_path, out, script=__file__, data_kind="measured",
                     inputs=[events_path], notes="synthetic-token training load; real compute on the device named")
    return out


class FakeLoad(GuardedThread):
    """SIMULATED load: drives FakeBackend activity and writes the same events."""

    def __init__(self, backend, events_path: str | Path, stop_event: threading.Event,
                 step_s: float = 0.25, ckpt_every: int = 40, ckpt_s: float = 2.0,
                 dip_activity: float = 0.15) -> None:
        super().__init__("fake-load", stop_event)
        self.backend = backend
        self.events_path = events_path
        self.step_s = step_s
        self.ckpt_every = ckpt_every
        self.ckpt_s = ckpt_s
        self.dip = dip_activity

    def loop(self) -> None:
        ev = EventLog(self.events_path)
        step = 0
        try:
            self.backend.set_activity(1.0)
            ev.write("config", 0, self.ckpt_every)
            while not self.stop_event.is_set():
                if self.stop_event.wait(self.step_s):
                    break
                step += 1
                ev.write("step", step, 0.0)
                if step % self.ckpt_every == 0:
                    ev.write("ckpt_start", step, 0)
                    self.backend.set_activity(self.dip)
                    self.stop_event.wait(self.ckpt_s)
                    self.backend.set_activity(1.0)
                    ev.write("ckpt_end", step, 0)
        finally:
            self.backend.set_activity(0.0)
            ev.write("end", step, 0)
            ev.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic training load with periodic checkpoints.")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--events", required=True, help="events CSV path")
    ap.add_argument("--size", choices=sorted(SIZES), default="small")
    ap.add_argument("--ckpt-every", type=int, default=0, help="steps between checkpoints (0 = calibrate)")
    ap.add_argument("--ckpt-period-s", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stop-file", default=None, help="the load stops when this file exists")
    ap.add_argument("--summary", default=None, help="write a load summary JSON here")
    ap.add_argument("--describe", action="store_true", help="print parameter counts and exit")
    a = ap.parse_args(argv)
    if a.describe:
        for name in SIZES:
            print(f"{name}: {param_count(name)} parameters, config (d, layers, heads, seq, batch) = {SIZES[name]}")
        return 0
    cfg = LoadConfig(size=a.size, minutes=a.minutes, ckpt_every=a.ckpt_every,
                     ckpt_period_s=a.ckpt_period_s, device=a.device, stop_file=a.stop_file)
    out = run_load(cfg, a.events, summary_path=a.summary)
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
