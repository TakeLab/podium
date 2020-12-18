import pytest

from podium.preproc.tokenizers import get_tokenizer


def test_callable_tokenizer():
    assert get_tokenizer(str.split) is str.split


def test_split_tokenizer():
    text = "The quick brown fox jumps,over,the-lazy-dog"

    assert get_tokenizer("split")(text) == text.split()
    assert get_tokenizer("split-,")(text) == text.split(",")
    assert get_tokenizer("split--")(text) == text.split("-")


def test_missing_tokenizer():
    with pytest.raises(ValueError):
        get_tokenizer(1)
