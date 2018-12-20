import pytest
from takepod.preproc.stemmer.croatian_stemmer import CroatianStemmer


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
