import torch

from indiana_c import IndianaC, IndianaCConfig
from indiana_c.generation import generate_with_think
from unittest.mock import patch


def test_forward():
    config = IndianaCConfig(
        vocab_size=10, block_size=16, n_layer=2, n_head=2, n_embd=32
    )
    model = IndianaC(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    logits, loss = model(idx, idx)
    assert logits.shape == (1, 4, config.vocab_size)
    assert loss is not None


def test_generate():
    config = IndianaCConfig(
        vocab_size=10, block_size=16, n_layer=2, n_head=2, n_embd=32
    )
    model = IndianaC(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=2)
    assert out.shape[-1] == 6


def test_generate_with_think_parses_sections():
    config = IndianaCConfig(vocab_size=256)

    class DummyModel:
        def __init__(self, cfg):
            self.cfg = cfg

        def eval(self):
            pass

        def generate(self, idx, max_new_tokens):
            addition = torch.tensor(
                [[ord(c) % self.cfg.vocab_size for c in "abc</think>answer"]],
                dtype=torch.long,
            )
            return torch.cat((idx, addition), dim=1)

    class DummyMonitor:
        def log(self, prompt, text):
            pass

    with patch("indiana_c.generation.IndianaC", DummyModel), patch(
        "indiana_c.generation.SelfMonitor", lambda: DummyMonitor()
    ), patch("indiana_c.generation.quantize_2bit", lambda model: None):
        thoughts, final = generate_with_think("Q", max_new_tokens=20, config=config)
    assert thoughts == "abc"
    assert final == "answer"
