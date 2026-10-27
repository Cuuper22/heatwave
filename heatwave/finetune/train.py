"""Fine-tuning loop for Timestral: AdamW, linear warmup + cosine decay, gradient clipping, early stopping on a fixed
validation set, deterministic seeds, checkpoint + resume, Hub-format export of the best weights.

Loss: mean pinball loss over the 21 levels and the horizon in the ORIGINAL scale, divided per window by the context
RMS (see heatwave.finetune.model). The forward pass is the inference pass (``symmetric=True``: the forecast of the
series and of its negation are averaged, as at inference) or, cheaper, the single raw pass (``symmetric=False``).
Step 0 is validated first and saved as the initial ``best``: the exported weights are never worse on the validation set
than the starting weights.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .data import Batch, Mixture
from .model import (GradForecaster, context_rms, load_for_training, pinball_per_window, save_checkpoint, set_trainable)


@dataclass
class TrainConfig:
    base: str = "5M"                  # Hub id, size (5M, 28M, 123M, 209M) or a local checkpoint directory
    mode: str = "full"                # full | head | lora
    last_blocks: int = 2              # mode head: transformer layers trained besides the head
    lora_r: int = 8
    lora_alpha: float = 16.0
    lr: float = 5e-5
    head_lr_mult: float = 1.0         # learning-rate multiplier of the quantile head and the final norm
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 500
    min_lr_frac: float = 0.0
    batch_size: int = 16
    grad_clip: float = 1.0
    eval_every: int = 50
    patience: int = 4                 # validations in a row without improvement before stopping
    min_delta: float = 0.0
    symmetric: bool = True
    seed: int = 0
    device: str = "cpu"
    max_seconds: Optional[float] = None
    micro_batch: Optional[int] = None  # windows per forward/backward chunk (gradient accumulation; same gradient as one batch)
    halt_after: Optional[int] = None  # stop (and save the resume state) after this many steps; the schedule still spans max_steps
    checkpoint_every: int = 100       # steps between resume-state saves (0: never)
    log_every: int = 10
    tf32: bool = True

    def to_json(self) -> dict:
        return asdict(self)


def lr_factor(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return cfg.min_lr_frac + (1.0 - cfg.min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))


def batch_loss(gf: GradForecaster, b: Batch, symmetric: bool) -> torch.Tensor:
    """Per-window normalised pinball loss ``(B,)`` of one batch."""
    H = b.future.shape[1]
    q = gf.quantiles(b.context, H, known_covariates=b.known, symmetric_averaging=symmetric)
    y = torch.from_numpy(b.future).to(q.device)
    rms = context_rms(torch.from_numpy(b.context)).to(q.device)
    return pinball_per_window(q, y, gf.quantile_levels, rms)


def slice_batch(b: Batch, lo: int, hi: int) -> Batch:
    return Batch(b.task, b.geometry, b.groups, b.context[lo:hi], b.future[lo:hi], None if b.known is None else b.known[lo:hi],
                 b.n_const_rows_zeroed)


@torch.no_grad()
def validate(gf: GradForecaster, batches: Sequence[Batch], symmetric: bool) -> dict:
    """Macro average over the validation batches (each batch: one task, geometry and covariate mode) of the mean loss."""
    per, labels = [], []
    for b in batches:
        try:
            loss = batch_loss(gf, b, symmetric).mean().item()
        except ValueError:
            loss = float("nan")
        per.append(loss)
        labels.append(f"{b.task}|{b.geometry[0]}x{b.geometry[1]}|{'+'.join(b.groups) or 'none'}")
    ok = [x for x in per if np.isfinite(x)]
    return {"macro": float(np.mean(ok)) if ok else float("nan"), "per_batch": dict(zip(labels, per)), "n_failed": len(per) - len(ok)}


def _param_groups(model: torch.nn.Module, cfg: TrainConfig):
    head_ids = {id(p) for p in list(model.head.parameters()) + list(model.transformer.out_norm.parameters())}
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        decay = p.ndim >= 2
        key = (is_head, decay)
        groups.setdefault(key, []).append(p)
    out = []
    for (is_head, decay), ps in sorted(groups.items()):
        out.append({"params": ps, "weight_decay": cfg.weight_decay if decay else 0.0,
                    "lr": cfg.lr * (cfg.head_lr_mult if is_head else 1.0), "is_head": is_head})
    return out


def _rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def train(cfg: TrainConfig, mixture: Mixture, val_batches: Sequence[Batch], out_dir: str | Path,
          log: Callable[[str], None] = print, resume: bool = True, provenance: Optional[dict] = None) -> dict:
    """Run the fine-tune. Writes ``out_dir/best`` (Hub format), ``out_dir/history.json`` and ``out_dir/train_state.pt``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if cfg.device.startswith("cuda") and cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(cfg.seed)

    gf = load_for_training(cfg.base, device=cfg.device)
    model = gf.model
    info = set_trainable(model, cfg.mode, last_blocks=cfg.last_blocks, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, seed=cfg.seed)
    model.to(cfg.device)
    opt = torch.optim.AdamW(_param_groups(model, cfg), betas=(0.9, 0.999), eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]

    step, best, best_step, bad, hist = 0, float("inf"), -1, 0, []
    n_skipped = 0
    state_path = out / "train_state.pt"
    elapsed_before = 0.0
    if resume and state_path.exists():
        st = torch.load(state_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, best, best_step, bad, hist = st["step"], st["best"], st["best_step"], st["bad"], st["hist"]
        n_skipped, elapsed_before = st.get("n_skipped", 0), st.get("elapsed", 0.0)
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"].cpu())
        log(f"resumed from step {step} (best {best:.5f} at step {best_step})")

    def save_state() -> None:
        tmp = state_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "best_step": best_step,
                    "bad": bad, "hist": hist, "n_skipped": n_skipped, "elapsed": elapsed_before + (time.time() - t_start),
                    "np_rng": _rng_state(rng), "torch_rng": torch.get_rng_state()}, tmp)
        tmp.replace(state_path)

    def do_validate(at_step: int) -> bool:
        nonlocal best, best_step, bad
        model.eval()
        v = validate(gf, val_batches, cfg.symmetric)
        improved = v["macro"] < best - cfg.min_delta
        hist.append({"step": at_step, "val_macro": v["macro"], "per_batch": v["per_batch"], "improved": bool(improved)})
        if improved:
            best, best_step, bad = v["macro"], at_step, 0
            save_checkpoint(model, out / "best", base_source=cfg.base,
                            extra={"step": at_step, "val_macro": v["macro"], "train_config": cfg.to_json(), **(provenance or {})})
        else:
            bad += 1
        log(f"step {at_step}: val {v['macro']:.5f} (best {best:.5f} @ {best_step}){' *' if improved else ''}")
        return improved

    t_start = time.time()
    if step == 0 and not hist:
        do_validate(0)
        save_state() if cfg.checkpoint_every else None
    stop_reason = "max_steps"
    ema = None
    while step < cfg.max_steps:
        if bad >= cfg.patience:
            stop_reason = "early_stopping"
            break
        if cfg.max_seconds is not None and elapsed_before + (time.time() - t_start) > cfg.max_seconds:
            stop_reason = "max_seconds"
            break
        if cfg.halt_after is not None and step >= cfg.halt_after:
            stop_reason = "halted"
            break
        factor = lr_factor(step, cfg)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * factor
        b = mixture.next(rng, cfg.batch_size)
        model.train()
        opt.zero_grad(set_to_none=True)
        mb = cfg.micro_batch or b.size
        try:
            total = 0.0
            for i in range(0, b.size, mb):
                part = batch_loss(gf, slice_batch(b, i, min(b.size, i + mb)), cfg.symmetric).sum() / b.size
                part.backward()
                total += part.item()
            loss = torch.tensor(total)
        except ValueError as e:     # degenerate window: the model rejects NaN output
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            log(f"step {step}: skipped batch ({e})")
            step += 1
            continue
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg.grad_clip)
        if not (torch.isfinite(loss) and torch.isfinite(gnorm)):
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            log(f"step {step}: non-finite loss or gradient, skipped")
        else:
            opt.step()
            ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        step += 1
        if cfg.log_every and step % cfg.log_every == 0:
            log(f"step {step}/{cfg.max_steps} loss {loss.item():.4f} ema {ema if ema is not None else float('nan'):.4f} "
                f"gnorm {float(gnorm):.3f} lr {opt.param_groups[0]['lr']:.2e} task {b.task} {b.geometry} {'+'.join(b.groups) or 'none'}")
        if cfg.eval_every and step % cfg.eval_every == 0:
            do_validate(step)
            if cfg.checkpoint_every:
                save_state()
        elif cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            save_state()
    else:
        if hist and hist[-1]["step"] != step:
            do_validate(step)
    if stop_reason == "max_steps" and bad >= cfg.patience:
        stop_reason = "early_stopping"
    if cfg.checkpoint_every:
        save_state()
    elapsed = elapsed_before + (time.time() - t_start)
    result = {
        "config": cfg.to_json(), "trainable": info, "steps_run": step, "best_step": best_step, "best_val_macro": best,
        "initial_val_macro": hist[0]["val_macro"] if hist else None, "stop_reason": stop_reason,
        "skipped_batches": n_skipped, "seconds": elapsed, "history": hist, "best_dir": str(out / "best"),
        "mixture": mixture.describe(),
        "device": cfg.device,
    }
    (out / "history.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    return result
