from indiana_core import generate_code, tokenizer


def test_tokenizer_handles_syntax_tokens() -> None:
    tokens = tokenizer.encode("{}=")
    assert tokens.shape[1] == 3
    decoded = tokenizer.decode(tokens)
    assert "{" in decoded and "}" in decoded and "=" in decoded


def test_generate_code_infill_returns_string() -> None:
    result = generate_code("a =", " b", max_new_tokens=2)
    assert isinstance(result, str)
    assert len(result) > 0
