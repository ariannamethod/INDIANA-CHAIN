import torch

from indiana_c import IndianaC, IndianaCConfig
from indiana_c.tokenizer import tokenizer


def test_forward():
    config = IndianaCConfig(block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = IndianaC(config)
    idx = tokenizer.encode("hello")
    logits, loss = model(idx, idx)
    assert logits.shape == (1, idx.shape[1], config.vocab_size)
    assert loss is not None


def test_generate():
    config = IndianaCConfig(block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = IndianaC(config)
    idx = tokenizer.encode("hello")
    out = model.generate(idx, max_new_tokens=2)
    assert out.shape[-1] == idx.shape[1] + 2


def test_generate_top_k_top_p():
    config = IndianaCConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, vocab_size=3)
    model = IndianaC(config)

    def fake_forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        logits = torch.tensor([1.0, 0.5, 0.1]).repeat(B, T, 1)
        return logits, None

    model.forward = fake_forward.__get__(model, IndianaC)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out_topk = model.generate(idx, max_new_tokens=2, top_k=1)
    assert (out_topk[0, 1:] == 0).all()
    out_topp = model.generate(idx, max_new_tokens=2, top_p=0.5)
    assert (out_topp[0, 1:] == 0).all()
