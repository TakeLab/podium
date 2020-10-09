import pytest
import spacy

from nltk.tokenize.toktok import ToktokTokenizer

from podium.preproc.tokenizers import get_tokenizer


TEST_STR = "The quick brown fox jumps over the lazy dog."


def test_callable_tokenizer():
    assert get_tokenizer(str.split) is str.split


def test_split_tokenizer():
    assert get_tokenizer("split")(TEST_STR) == TEST_STR.split()


def test_spacy_tokenizer():
    spacy_tokenizer = spacy.load("en")
    spacy_tokens = [token.text for token in spacy_tokenizer.tokenizer(TEST_STR)]
    assert get_tokenizer("spacy")(TEST_STR) == spacy_tokens


def test_moses_tokenizer():
    sacremoses = pytest.importorskip("sacremoses")
    assert get_tokenizer("moses")(TEST_STR) == sacremoses.MosesTokenizer().tokenize(TEST_STR)


def test_toktok_tokenizer():
    assert get_tokenizer("toktok")(TEST_STR) == ToktokTokenizer().tokenize(TEST_STR)


def test_missing_dataset():
    with pytest.raises(ValueError):
        get_tokenizer(1)
