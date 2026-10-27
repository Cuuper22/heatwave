"""Differentiable Timestral forecast pass, pinball loss, trainable-parameter modes and Hub-format export.

``TimestralForecaster.forecast`` (timestral/forecaster.py) runs under ``torch.inference_mode``. ``GradForecaster``
subclasses it and re-uses its tensor builders (``_as_context``, ``_as_covariates``, ``_assemble_rows`` through
``_forecast_raw``: scaling, patching, masks, covariate rows, block decoding, refine frame, quantile cap) and
only differs in (a) running with autograd and (b) holding the refine-frame statistics constant for the backward pass
(the forward values are unchanged). With ``symmetric_averaging=True`` and ``clip_nonnegative=True`` its output equals
``TimestralForecaster.forecast`` exactly (tests/test_finetune.py).

Loss: mean pinball loss over the 21 quantile levels and the horizon, computed in the ORIGINAL scale of the series
(the scale the evaluation scores use), divided per window by the root-mean-square of the observed context so that
series of different units (W, deg C, USD/MWh) carry comparable weight. WQL (heatwave.forecast.metrics.wql) divides the
pinball loss by the mean absolute actual of a dataset; this per-window division is the training analogue.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from timestral import TimestralForecaster as _BaseForecaster
from timestral.loading import CONFIG_FILE, WEIGHTS_FILE, load_model, resolve_checkpoint
from timestral.scaler import advancing_block_frames, clamp_nonfinite

HUB_IDS = {"5M": "mistralai/Timestral-5M", "28M": "mistralai/Timestral-28M",
           "123M": "mistralai/Timestral-123M", "209M": "mistralai/Timestral-209M"}
LICENSE_NOTE = (
    "This directory holds a modified copy of a Timestral model (Mistral AI, Modified MIT License, attribution notice "
    "2026 - Mistral AI). The upstream LICENSE file is copied here unchanged and applies to these weights. "
    "Restriction in the upstream license: no rights are granted to a company (or employer) whose global consolidated "
    "monthly revenue exceeded $20 million in the preceding month; this applies to derivatives such as these weights."
)


# ------------------------------------------------------------------------------------------------ forward pass
class GradForecaster(_BaseForecaster):
    """Timestral forecaster whose forward pass keeps the autograd graph."""

    def _refine_block(self, quantiles: Tensor, context: Tensor, mask: Tensor, series_rows: Optional[Tensor]) -> Tensor:
        """Same values as the parent; the advancing-frame statistics are constants in the backward pass."""
        ps = self.patch_size
        n_pred = quantiles.shape[1] // ps
        if n_pred <= 1:
            return quantiles
        with torch.no_grad():
            _, loc, scale = self.model.scale(context, mask)
            loc0, scale0 = loc[:, -1], scale[:, -1]
            median = quantiles[..., self.median_index].unflatten(-1, (n_pred, ps)).to(torch.float64)
            sinh_median = (median - loc0[:, None, None]) / scale0[:, None, None]
            loc_j, scale_j = advancing_block_frames(context, mask, sinh_median, self.config.minimum_scale)
            first = torch.zeros(loc_j.shape, dtype=torch.bool, device=loc_j.device)
            first[:, 0] = True
            loc_j = torch.where(first, loc0[:, None], loc_j.to(quantiles.dtype)).repeat_interleave(ps, dim=-1)
            scale_j = torch.where(first, scale0[:, None], scale_j.to(quantiles.dtype)).repeat_interleave(ps, dim=-1)
        sinh_norm = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
        refined = clamp_nonfinite(sinh_norm * scale_j[..., None] + loc_j[..., None])
        if series_rows is not None:
            refined = torch.where(series_rows[:, None, None], refined, quantiles)
        return refined

    def quantiles(
        self,
        context,
        prediction_length: int,
        *,
        known_covariates=None,
        past_covariates=None,
        observation_mask=None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
    ) -> Tensor:
        """``(B, V, H, Q)`` quantiles, differentiable. Same arguments and processing as ``TimestralForecaster.forecast``."""
        assert prediction_length >= 1
        inputs, mask = self._as_context(context, observation_mask)
        b, t = inputs.shape[0], inputs.shape[-1]
        past = self._as_covariates(past_covariates, b, t, "past_covariates")
        known = self._as_covariates(known_covariates, b, t + prediction_length, "known_covariates")
        covariates = {"past_covariates": past, "known_covariates": known}
        mirrored = {k: (-v if v is not None else None) for k, v in covariates.items()}
        if known is not None and not self.supports_known_covariates:
            raise ValueError("this checkpoint cannot condition on known-future covariates")
        q = self._forecast_raw(inputs, mask, prediction_length, **covariates)
        if symmetric_averaging:
            qm = self._forecast_raw(-inputs, mask, prediction_length, **mirrored)
            q = 0.5 * (q + (-qm).flip(-1))
        if clip_nonnegative:
            from timestral.forecaster import nonnegative_context

            eligible = nonnegative_context(inputs, mask)
            q = torch.where(eligible[..., None, None], q.clamp_min(0), q)
        return q


def load_for_training(source: str | Path, device: str | torch.device = "cpu") -> GradForecaster:
    """``source`` is a Hub id (``mistralai/Timestral-5M``), a size (``5M``) or a local checkpoint directory."""
    s = str(source)
    if s in HUB_IDS:
        s = HUB_IDS[s]
    model = load_model(s, device=device)
    return GradForecaster(model)


# ------------------------------------------------------------------------------------------------ loss
def context_rms(context: Tensor) -> Tensor:
    """``(B, 1, 1)`` root-mean-square of the finite context values per window (1 if none, floor 1e-8)."""
    x = torch.as_tensor(context, dtype=torch.float32)
    x = x.reshape(x.shape[0], -1)
    ok = torch.isfinite(x)
    n = ok.sum(-1).clamp(min=1)
    ms = torch.where(ok, x, torch.zeros_like(x)).square().sum(-1) / n
    rms = ms.sqrt()
    rms = torch.where(ok.any(-1) & (rms > 1e-8), rms, torch.ones_like(rms))
    return rms.reshape(-1, 1, 1)


def pinball_per_window(q: Tensor, y: Tensor, levels: Sequence[float], rms: Tensor) -> Tensor:
    """Per-window mean pinball loss over horizon and levels, divided by ``rms``. ``q`` is (B, 1, H, Q), ``y`` is (B, H).

    Future steps that are not finite are left out of the mean."""
    lv = torch.as_tensor(levels, dtype=q.dtype, device=q.device)
    yy = y[:, None, :, None]
    ok = torch.isfinite(yy)
    diff = torch.where(ok, yy - q, torch.zeros_like(q))
    loss = torch.maximum(lv * diff, (lv - 1.0) * diff) * ok
    denom = ok.expand_as(q).sum(dim=(1, 2, 3)).clamp(min=1)
    per = loss.sum(dim=(1, 2, 3)) / denom
    return per / rms.reshape(-1).to(per.dtype)


# ------------------------------------------------------------------------------------------------ LoRA (hand-written)
class LoRALinear(nn.Module):
    """``y = base(x) + (alpha / r) * B(A(x))`` with A: in -> r, B: r -> out; B starts at zero, so the module starts as the base."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, seed: int = 0) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = float(alpha) / float(r)
        g = torch.Generator().manual_seed(seed)
        a = torch.empty(r, base.in_features)
        bound = 1.0 / math.sqrt(base.in_features)
        a.uniform_(-bound, bound, generator=g)
        self.lora_a = nn.Parameter(a.to(base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling

    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            w = self.base.weight + self.scaling * (self.lora_b @ self.lora_a)
            lin.weight.copy_(w)
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin.to(device=self.base.weight.device, dtype=self.base.weight.dtype)


def _lora_targets(layer: nn.Module) -> Iterable[tuple[nn.Module, str]]:
    for attn in layer.attns:
        yield attn, "wQKV"
        yield attn, "wO"
        if attn._gate is not None:
            yield attn, "_gate"
    yield layer.mlp, "w12"
    yield layer.mlp, "w3"


def add_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, last_blocks: Optional[int] = None, seed: int = 0) -> int:
    """Wrap every linear map of the transformer layers (the last ``last_blocks`` layers if given). Returns the number wrapped."""
    layers = model.transformer.layers
    first = 0 if last_blocks is None else max(0, len(layers) - last_blocks)
    n = 0
    for i, layer in enumerate(layers):
        if i < first:
            continue
        for owner, name in list(_lora_targets(layer)):
            base = getattr(owner, name)
            if isinstance(base, LoRALinear):
                continue
            setattr(owner, name, LoRALinear(base, r, alpha, seed=seed + 1000 * i + n))
            n += 1
    return n


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear by a plain Linear holding the merged weight (the layout of the Hub checkpoint)."""
    n = 0
    for layer in model.transformer.layers:
        for owner, name in list(_lora_targets(layer)):
            m = getattr(owner, name)
            if isinstance(m, LoRALinear):
                setattr(owner, name, m.merged())
                n += 1
    return n


# ------------------------------------------------------------------------------------------------ trainable parameters
MODES = ("full", "head", "lora")


def set_trainable(model: nn.Module, mode: str, *, last_blocks: int = 2, lora_r: int = 8, lora_alpha: float = 16.0,
                  seed: int = 0) -> dict:
    """Choose which parameters train. Returns a description with parameter counts.

    full  every parameter.
    head  the quantile head, the final norm and the last ``last_blocks`` transformer layers (the rest frozen).
    lora  LoRA adapters (rank ``lora_r``) on every linear map of the transformer layers, plus the quantile head.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    total = sum(p.numel() for p in model.parameters())
    info: dict = {"mode": mode}
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "head":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
        for p in model.transformer.out_norm.parameters():
            p.requires_grad_(True)
        layers = model.transformer.layers
        for layer in layers[max(0, len(layers) - last_blocks):]:
            for p in layer.parameters():
                p.requires_grad_(True)
        info["last_blocks"] = last_blocks
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        n = add_lora(model, lora_r, lora_alpha, seed=seed)
        for p in model.head.parameters():
            p.requires_grad_(True)
        info.update({"lora_rank": lora_r, "lora_alpha": lora_alpha, "lora_modules": n})
    info["n_params_total_base"] = total
    info["n_params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return info


# ------------------------------------------------------------------------------------------------ export / import
def save_checkpoint(model: nn.Module, out_dir: str | Path, *, base_source: str | Path, extra: Optional[dict] = None) -> Path:
    """Write ``config.json`` + ``model.safetensors`` (the Hub layout) plus the upstream LICENSE and a short README.

    LoRA adapters are merged into a copy first, so the saved state dict has the standard keys and
    ``TimestralForecaster.from_pretrained(out_dir)`` loads it unchanged. The model passed in is left as it is."""
    import copy

    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = model
    if any(isinstance(x, LoRALinear) for x in model.modules()):
        m = copy.deepcopy(model)
        merge_lora(m)
    sd = {k: v.detach().to("cpu").contiguous() for k, v in m.state_dict().items()}
    save_file(sd, str(out / WEIGHTS_FILE))
    m.config.save(out / CONFIG_FILE)
    src = resolve_checkpoint(HUB_IDS.get(str(base_source), str(base_source)))
    if (src / "LICENSE").exists():
        shutil.copyfile(src / "LICENSE", out / "LICENSE")
    (out / "README.md").write_text(
        f"# Fine-tuned Timestral checkpoint\n\nBase model: {base_source}\n\n{LICENSE_NOTE}\n"
        + ("\n```json\n" + json.dumps(extra, indent=1, default=str) + "\n```\n" if extra else "")
    )
    if extra:
        (out / "finetune.json").write_text(json.dumps(extra, indent=1, default=str) + "\n")
    return out
