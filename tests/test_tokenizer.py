from indiana_core import tokenizer


def has_token(token: str) -> bool:
    return tokenizer.token_to_id(token) is not None


def word_in_vocab(word: str) -> bool:
    return has_token(word) or has_token("Ġ" + word)


def test_basic_chars_present():
    assert has_token("a")
    assert has_token("b")


def test_dataset_words_present():
    assert word_in_vocab("Mary")
    assert word_in_vocab("apples")
