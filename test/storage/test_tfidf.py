import numpy as np
from takepod.storage.vocab import Vocab
from takepod.storage.tfidf import TfIdfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


DATA = [
    'this is the first document',
    'this document is the second document',
    'and this is the third one',
    'is this the first document'
]


NUMERICALIZED_DATA = [
    np.array([0, 1, 2, 4, 3]),
    np.array([0, 3, 1, 2, 5, 3]),
    np.array([6, 0, 1, 2, 7, 8]),
    np.array([1, 0, 2, 4, 3])]


def test_build_count_matrix_from_tensor():
    vocab = Vocab(specials=())
    for i in DATA:
        vocab += i.split(" ")
    vocab.finalize()
    tfidf = TfIdfVectorizer(vocab=vocab)
    numericalized_data = []
    for i in DATA:
        numericalized_data.append(vocab.numericalize(i.split(" ")))

    count_matrix = tfidf._build_count_matrix(data=numericalized_data,
                                             unpack_data=tfidf._get_tensor_values)

    expected = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 2, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0]])
    assert np.all(count_matrix == expected)


def test_tfidf_equality_with_scikit():
    scikit_vectorizer = TfIdfVectorizer()
    