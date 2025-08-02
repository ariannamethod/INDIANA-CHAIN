"""Naive 2-bit quantization helpers for Indiana-C."""
from __future__ import annotations

import torch
import torch.nn as nn


def _quantize_tensor_2bit(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to 2-bit signed values.

    Returns a tuple of (quantized_tensor, scale).
    """
    max_val = t.abs().max()
    scale = max_val / 1.5 + 1e-8
    q = torch.clamp((t / scale).round(), -2, 1).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32, device=t.device)


class Linear2Bit(nn.Module):
    """Linear layer with weights stored in 2-bit form."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        q_w, scale = _quantize_tensor_2bit(linear.weight.data)
        self.register_buffer("weight_q", q_w)
        self.register_buffer("scale", scale)
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w = self.weight_q.float() * self.scale
        return torch.nn.functional.linear(x, w, self.bias)


def quantize_model_2bit(module: nn.Module) -> nn.Module:
    """Recursively quantize Linear submodules to 2-bit weights."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, Linear2Bit(child))
        else:
            quantize_model_2bit(child)
    return module
