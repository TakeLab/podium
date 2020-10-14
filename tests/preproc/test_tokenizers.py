import pytest

from podium.preproc.tokenizers import get_tokenizer


def test_callable_tokenizer():
    assert get_tokenizer(str.split) is str.split


def test_missing_tokenizer():
    with pytest.raises(ValueError):
        get_tokenizer(1)
