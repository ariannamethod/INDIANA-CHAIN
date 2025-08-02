import torch

from .model import IndianaC, IndianaCConfig


def encode(text: str, vocab_size: int) -> torch.Tensor:
    return torch.tensor([[ord(c) % vocab_size for c in text]], dtype=torch.long)


def decode(tokens: torch.Tensor) -> str:
    return "".join(chr(int(t)) for t in tokens)


def generate_text(
    prompt: str,
    max_new_tokens: int = 50,
    config: IndianaCConfig | None = None,
) -> str:
    config = config or IndianaCConfig()
    model = IndianaC(config)
    model.eval()
    idx = encode(prompt, config.vocab_size)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    return decode(out[0])
