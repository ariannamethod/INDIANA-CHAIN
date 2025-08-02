from pathlib import Path

import torch

from .model import IndianaC, IndianaCConfig
from .monitor import SelfMonitor
from .quantize import quantize_2bit
from .logger import (
    estimate_complexity_and_entropy,
    thought_logger,
)

CORE_PROMPT = (
    Path(__file__).resolve().parent.parent / "core_prompt.txt"
).read_text(encoding="utf-8")
print("core_prompt.txt loaded [OK]")


def encode(text: str, vocab_size: int) -> torch.Tensor:
    return torch.tensor([[ord(c) % vocab_size for c in text]], dtype=torch.long)


def decode(tokens: torch.Tensor) -> str:
    return "".join(chr(int(t)) for t in tokens)


def generate_text(
    prompt: str | None = None,
    max_new_tokens: int = 50,
    config: IndianaCConfig | None = None,
    *,
    log_reasoning: bool = False,
) -> str | tuple[str, dict[str, float | int]]:
    prompt = prompt or CORE_PROMPT
    config = config or IndianaCConfig()
    model = IndianaC(config)
    quantize_2bit(model)
    monitor = SelfMonitor()
    model.eval()
    idx = encode(prompt, config.vocab_size)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    text = decode(out[0])
    monitor.log(prompt, text)
    complexity, entropy = estimate_complexity_and_entropy(text)
    record = thought_logger.log_turn(text, complexity, entropy)
    if log_reasoning:
        return text, {"complexity": record.complexity, "entropy": record.entropy, "timestamp": record.timestamp}
    return text
