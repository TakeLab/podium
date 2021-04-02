import inspect

import pytest

from podium.datasets import ExampleFactory
from podium.field import Field
from podium.preproc.hooks import (
    HookType,
    KeywordExtractor,
    MosesNormalizer,
    NLTKStemmer,
    RegexReplace,
    SpacyLemmatizer,
    TextCleanUp,
    as_posttokenize_hook,
    remove_stopwords,
    truecase,
)


TEXT = """\
Sources tell us that Google is acquiring Kaggle, a platform that hosts
data science and machine learning competitions. Details about the
transaction remain somewhat vague, but given that Google is hosting
its Cloud Next conference in San Francisco this week, the official
announcement could come as early as tomorrow.  Reached by phone,
Kaggle co-founder CEO Anthony Goldbloom declined to deny that the
acquisition is happening. Google itself declined
'to comment on rumors'.
"""


@pytest.mark.require_package("spacy")
@pytest.mark.require_spacy_model("en_core_web_sm")
def test_remove_stopwords():
    data = "I'll tell you a joke"
    field = Field(name="data")
    field.add_posttokenize_hook(remove_stopwords("en"))
    example = ExampleFactory([field]).from_list([data])

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
            marks=[
                pytest.mark.require_package("spacy"),
                pytest.mark.require_spacy_model("en_core_web_sm"),
            ],
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


@pytest.mark.parametrize(
    "alg,alg_pkg_name",
    [
        ("yake", "yake"),
        ("rake", "rake_nltk"),
    ],
)
def test_keyword_extractor(alg, alg_pkg_name):
    pytest.importorskip(alg_pkg_name)

    field = Field(name="data", tokenizer=None, keep_raw=True)
    field.add_posttokenize_hook(KeywordExtractor(alg))
    example = ExampleFactory([field]).from_list([TEXT])

    # make sure all the keywords originate from the raw data
    text_ = TEXT.lower()
    assert all(kw in text_ for kws in example["data"][1] for kw in kws.lower().split())

@pytest.mark.require_package("yake")
@pytest.mark.require_package("spacy")
@pytest.mark.require_spacy_model("en_core_web_sm")
def test_hook_type():
    pretokenize_hooks = [
        MosesNormalizer(),
        RegexReplace([("", "")]),
        TextCleanUp(),
        truecase(),
    ]
    posttokenize_hooks = [
        remove_stopwords("en"),
        SpacyLemmatizer(),
        NLTKStemmer(),
        KeywordExtractor("yake"),
    ]

    assert all([hook.__hook_type__ == HookType.PRETOKENIZE for hook in pretokenize_hooks])
    assert all(
        [hook.__hook_type__ == HookType.POSTTOKENIZE for hook in posttokenize_hooks]
    )


def test_hook_conversion():
    field = Field(name="data", tokenizer="split", keep_raw=True)
    text_clean_up_hook = TextCleanUp(replace_url="<URL>")

    assert text_clean_up_hook.__hook_type__ == HookType.PRETOKENIZE
    with pytest.raises(ValueError):
        field.add_posttokenize_hook(text_clean_up_hook)

    text_clean_up_hook = as_posttokenize_hook(text_clean_up_hook)
    assert text_clean_up_hook.__hook_type__ == HookType.POSTTOKENIZE

    field.add_posttokenize_hook(text_clean_up_hook)

    data = "url to github is https://github.com"
    example = ExampleFactory([field]).from_list([data])

    assert example["data"][1] == ["url", "to", "github", "is", "<URL>"]
