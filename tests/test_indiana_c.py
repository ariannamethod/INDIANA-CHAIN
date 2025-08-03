from indiana_core import IndianaC, IndianaCConfig, tokenizer


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
