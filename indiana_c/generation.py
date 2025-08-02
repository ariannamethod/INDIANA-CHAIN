from pathlib import Path

import torch

from .model import IndianaC, IndianaCConfig
from .monitor import SelfMonitor
from .quantize import quantize_2bit

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
) -> str:
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
    return text


def generate_with_think(
    prompt: str | None = None,
    max_new_tokens: int = 50,
    config: IndianaCConfig | None = None,
) -> tuple[str, str]:
    prompt = prompt or CORE_PROMPT
    config = config or IndianaCConfig()
    model = IndianaC(config)
    quantize_2bit(model)
    monitor = SelfMonitor()
    model.eval()
    full_prompt = f"{prompt} <think>"
    idx = encode(full_prompt, config.vocab_size)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    text = decode(out[0])
    monitor.log(full_prompt, text)
    if "</think>" in text:
        start = text.find("<think>") + len("<think>")
        end = text.find("</think>")
        thoughts = text[start:end].strip()
        final_answer = text[end + len("</think>") :].strip()
    else:
        thoughts = ""
        final_answer = text
    return thoughts, final_answer
