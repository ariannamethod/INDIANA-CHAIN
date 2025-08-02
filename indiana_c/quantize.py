"""Weight quantization utilities for Indiana-C."""

from __future__ import annotations

import torch
from torch import nn

_SUPPORTED_BITS = {2, 4, 8}


@torch.no_grad()
def quantize(model: nn.Module, bits: int) -> None:
    """Quantize the model weights to the specified bit precision in-place.

    Args:
        model: The module whose parameters will be quantized.
        bits: Target bit precision. Supported values are 2, 4 and 8.
    """
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"Unsupported bit-width: {bits}")

    max_q = 2 ** (bits - 1) - 1
    for param in model.parameters():
        if param.dtype not in (torch.float32, torch.float64):
            continue
        max_val = param.abs().max()
        if max_val == 0:
            continue
        scale = max_val / max_q
        q = (param / scale).round().clamp(-max_q, max_q)
        param.copy_(q * scale)


__all__ = ["quantize"]
