import torch

from indiana_c import IndianaC, IndianaCConfig, quantize_model_2bit


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


def test_quantized_generate():
    config = IndianaCConfig(
        vocab_size=10, block_size=16, n_layer=2, n_head=2, n_embd=32
    )
    model = IndianaC(config)
    quantize_model_2bit(model)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=2)
    assert out.shape[-1] == 6
