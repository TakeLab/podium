import pytest
from takepod.preproc.stemmer.croatian_stemmer import (
    CroatianStemmer,
    get_croatian_stemmer_hook)


@pytest.fixture(scope='module')
def cro_stemmer():
    yield CroatianStemmer()


def test_croatian_stemmer_stem_word(cro_stemmer):
    assert cro_stemmer.stem_word('babicama') == 'babic'
    assert cro_stemmer.stem_word('babice') == 'babic'


def test_croatian_stemmer_stem_nostem_word(cro_stemmer):
    assert cro_stemmer.stem_word('želimo') == 'želimo'
    assert cro_stemmer.stem_word('jesmo') == 'jesmo'


def test_croatian_stemmer_no_vowel_word(cro_stemmer):
    # this is an actual word in Croatian, look it up
    assert cro_stemmer.stem_word('sntntn') == 'sntntn'


def test_croatian_stemmer_transformative_word(cro_stemmer):
    assert cro_stemmer.stem_word('turizama') == 'turizm'


def test_croatian_stemmer_preserves_case(cro_stemmer):
    assert cro_stemmer.stem_word('Turizam') == 'Turizm'


@pytest.mark.parametrize(
    "example_raw, example_words, expected_result",
    [
        ("babice turizama", ["babice", "turizama"], ["babic", "turizm"]),
        ("jesmo sntntn turizama", ["jesmo", "sntntn", "turizama"],
         ["jesmo", "sntntn", "turizm"]),
    ]
)
def test_croatian_stemmer_hook(example_raw, example_words, expected_result):
    stemmer_hook = get_croatian_stemmer_hook()
    result_raw, result_tokenized = stemmer_hook(
        raw=example_raw,
        tokenized=example_words)
    assert result_tokenized == expected_result
    assert result_raw == example_raw


@pytest.mark.parametrize(
    "example_words, expected_result",
    [
        (["babice", "turizama"], ["babic", "turizm"]),
        (["jesmo", "sntntn"], ["jesmo", "sntntn"]),
    ]
)
def test_croatian_stemmer_hook_raw_none(example_words, expected_result):
    stemmer_hook = get_croatian_stemmer_hook()
    result_raw, result_tokenized = stemmer_hook(
        raw=None,
        tokenized=example_words)
    assert result_tokenized == expected_result
    assert result_raw is None
