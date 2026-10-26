"""Timestral wrapper with the shared QuantileForecast interface.

Timestral (Mistral, Oct 2026) is a family of open time-series foundation
models: 5M/28M/123M/209M parameters, up to 32,768 context steps, 21 output
quantiles, past-only and known-future covariates. Code:
https://github.com/mistralai/timestral (Apache 2.0). Weights:
https://huggingface.co/mistralai (modified MIT).

Measured on this repo's 2-core CPU VM (context 4096, horizon 256, one series):
5M about 0.04 s, 28M about 0.09 s, 209M about 0.6 s per forecast.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import QUANTILE_LEVELS, QuantileForecast, as_bvt

SIZES = ("5M", "28M", "123M", "209M")


class TimestralForecaster:
    """Zero-shot (or fine-tuned, via ``weights_dir``) Timestral."""

    def __init__(
        self,
        size: str = "28M",
        *,
        weights_dir: Optional[str] = None,
        device: str = "cpu",
        threads: Optional[int] = None,
    ) -> None:
        import torch
        from timestral import TimestralForecaster as _TF

        if threads is not None:
            torch.set_num_threads(threads)
        if weights_dir is None and size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}")
        source = weights_dir or f"mistralai/Timestral-{size}"
        self._model = _TF.from_pretrained(source, device=device)
        self.size = size
        self.name = f"timestral-{size}" + ("-finetuned" if weights_dir else "")
        if tuple(round(q, 2) for q in self._model_levels()) != QUANTILE_LEVELS:
            raise RuntimeError("Timestral quantile grid differs from QUANTILE_LEVELS")

    def _model_levels(self) -> tuple[float, ...]:
        x = np.zeros((1, 1, 64))
        x[0, 0] = np.arange(64)
        return tuple(self._model.forecast(x, prediction_length=1).quantile_levels)

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        *,
        known_covariates: Optional[np.ndarray] = None,
        past_covariates: Optional[np.ndarray] = None,
    ) -> QuantileForecast:
        ctx = as_bvt(context)
        kwargs = {}
        if known_covariates is not None:
            kc = np.asarray(known_covariates, dtype=np.float64)
            if kc.shape[0] != ctx.shape[0] or kc.shape[-1] != ctx.shape[-1] + horizon:
                raise ValueError(
                    f"known_covariates must be (B, C, T+H); got {kc.shape} for "
                    f"context {ctx.shape} and horizon {horizon}"
                )
            kwargs["known_covariates"] = kc
        if past_covariates is not None:
            pc = np.asarray(past_covariates, dtype=np.float64)
            if pc.shape[0] != ctx.shape[0] or pc.shape[-1] != ctx.shape[-1]:
                raise ValueError("past_covariates must be (B, C, T)")
            kwargs["past_covariates"] = pc
        out = self._model.forecast(ctx, prediction_length=horizon, **kwargs)
        q = out.quantiles
        q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
        return QuantileForecast(values=q.astype(np.float64), method=self.name)


def default_threads() -> int:
    return max(1, (os.cpu_count() or 1))
