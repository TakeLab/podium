from takepod.preproc.stemmer.croatian_stemmer import CroatianStemmer
import pytest


@pytest.fixture(scope='module')
def cro_stemmer():
    yield CroatianStemmer()


def test_croatian_stemmer_stem_word(cro_stemmer):
    # stemmer needs refactoring, see it's source file
    assert cro_stemmer.korjenuj('babicama') == 'babic'
    assert cro_stemmer.korjenuj('babice') == 'babic'
