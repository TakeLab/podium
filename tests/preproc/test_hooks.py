import inspect

import pytest

from podium.field import Field
from podium.preproc.functional import remove_stopwords, truecase
from podium.preproc.hooks import (
    MosesNormalizer,
    NLTKStemmer,
    RegexReplace,
    SpacyLemmatizer,
    TextCleanUp,
)
from podium.storage import ExampleFactory

from ..util import run_spacy


@run_spacy
def test_remove_stopwords():
    data = "I'll tell you a joke"
    field = Field(name="data")
    field.add_posttokenize_hook(remove_stopwords("en"))
    example = ExampleFactory([field]).from_list(data)

    assert "you" not in example["data"][1]
    assert "a" not in example["data"][1]


def test_truecase():
    pytest.importorskip("truecase")

    data = "hey how are you"
    field = Field(name="data", tokenizer=None, keep_raw=True)
    field.add_pretokenize_hook(truecase())
    example = ExampleFactory([field]).from_list([data])

    assert "Hey how are you" == example["data"][0]


def test_moses_normalizer():
    pytest.importorskip("sacremoses")

    data = "What's    up!"
    field = Field(name="data", tokenizer=None, keep_raw=True)
    normalizer = MosesNormalizer()
    field.add_pretokenize_hook(normalizer)
    example = ExampleFactory([field]).from_list([data])

    assert "What's up!" == example["data"][1]


@pytest.mark.parametrize(
    "hook",
    [
        NLTKStemmer("en"),
        pytest.param(
            lambda: SpacyLemmatizer("en"),
            marks=run_spacy,
        ),
    ],
)
def test_lemmatization_and_stemming(hook):
    # we need this to postpone initialization
    # in pytest.mark.parametrize
    if inspect.isfunction(hook):
        hook = hook()

    data = "stemming playing books"
    field = Field(name="data")
    field.add_posttokenize_hook(hook)
    example = ExampleFactory([field]).from_list([data])

    # we don't check the exact results,
    # instead we expect some modifications
    assert data != example["data"][1]


def test_regex_replace():
    data = "This item costs 100$."
    field = Field(name="data", tokenizer=None, keep_raw=True)
    regex_replace = RegexReplace([(r"\d+", "<NUMBER>"), (r"\s+", "<WHITESPACE>")])
    field.add_pretokenize_hook(regex_replace)
    example = ExampleFactory([field]).from_list([data])

    expected_raw = "This<WHITESPACE>item<WHITESPACE>costs<WHITESPACE><NUMBER>$."
    assert expected_raw == example["data"][1]


@pytest.mark.parametrize(
    "kwargs,data,expected_output",
    [
        ({"remove_line_breaks": True}, "some data\n", "some data"),
        ({"remove_punct": True}, "hey!?", "hey"),
        ({"replace_url": "<URL>"}, "url: https://github.com", "url: <URL>"),
        ({"replace_email": "<EMAIL>"}, "email: john.doe@gmail.com", "email: <EMAIL>"),
        (
            {"replace_phone_number": "<PNUM>"},
            "reach me at 555-123-4567",
            "reach me at <PNUM>",
        ),
        ({"replace_number": "<NUM>"}, "10 + 3.5 = 13.5", "<NUM> + <NUM> = <NUM>"),
        ({"replace_digit": "<DIG>"}, "it's 2 am", "it's <DIG> am"),
        ({"replace_currency_symbol": "<CURR>"}, "this is 100$", "this is 100<CURR>"),
    ],
)
def test_text_clean_up(kwargs, data, expected_output):
    pytest.importorskip("cleantext")

    field = Field(name="data", tokenizer=None, keep_raw=True)
    field.add_pretokenize_hook(TextCleanUp(**kwargs))
    example = ExampleFactory([field]).from_list([data])

    assert expected_output == example["data"][1]
