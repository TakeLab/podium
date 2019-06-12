from test.storage.conftest import TABULAR_TEXT
import numpy as np
import pytest
from sklearn.feature_extraction import text
from takepod.storage.vocab import Vocab
from takepod.storage.tfidf import TfIdfVectorizer
from takepod.storage.field import Field


DATA = [
    'this is the first document',
    'this document is the second document',
    'and this is the third one',
    'is this the first document'
]


def get_numericalized_data(data, vocab):
    numericalized_data = []
    for i in data:
        numericalized_data.append(vocab.numericalize(i.split(" ")))
    return numericalized_data


def test_build_count_matrix_from_tensor():
    vocab = Vocab(specials=())
    for i in DATA:
        vocab += i.split(" ")
    vocab.finalize()
    tfidf = TfIdfVectorizer(vocab=vocab)

    numericalized_data = get_numericalized_data(data=DATA, vocab=vocab)
    count_matrix = tfidf._build_count_matrix(data=numericalized_data,
                                             unpack_data=tfidf._get_tensor_values)

    expected = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 2, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0]])
    assert np.all(count_matrix == expected)


@pytest.mark.usefixtures("tabular_dataset")
def test_tfidf_equality_with_scikit(tabular_dataset):
    tabular_dataset.finalize_fields()
    text_field = tabular_dataset.field_dict["text"]
    vocab = text_field.vocab

    tfidf = TfIdfVectorizer(vocab=vocab)
    tfidf.fit(dataset=tabular_dataset, field=text_field)

    numericalized_data = get_numericalized_data(data=TABULAR_TEXT, vocab=vocab)
    vectorized_text = tfidf.transform(numericalized_data)

    scikit_vectorizer = text.TfidfVectorizer(vocabulary=vocab.stoi,
                                             token_pattern=r"(?u)\b\w+\b")
    scikit_vectorizer.fit(TABULAR_TEXT)
    scikit_vectors = scikit_vectorizer.transform(TABULAR_TEXT)

    assert np.allclose(a=vectorized_text.todense(),
                       b=scikit_vectors.todense(),
                       rtol=0, atol=1.e-6)


def test_fit_dataset_none_error():
    tfidf = TfIdfVectorizer()
    with pytest.raises(ValueError):
        tfidf.fit(dataset=None, field=Field("text"))


@pytest.mark.usefixtures("tabular_dataset")
def test_fit_field_none_error(tabular_dataset):
    tfidf = TfIdfVectorizer()
    with pytest.raises(ValueError):
        tfidf.fit(dataset=tabular_dataset, field=None)


@pytest.mark.usefixtures("tabular_dataset")
def test_fit_invalid_field_error(tabular_dataset):
    tfidf = TfIdfVectorizer()
    with pytest.raises(ValueError):
        tfidf.fit(dataset=tabular_dataset, field=Field("non_present_field"))


def test_transform_before_fit_error():
    tfidf = TfIdfVectorizer()
    with pytest.raises(RuntimeError):
        tfidf.transform([[1, 1, 1, 1, 1, 0, 0, 0, 0]])


@pytest.mark.usefixtures("tabular_dataset")
def test_vocab_none(tabular_dataset):
    tfidf = TfIdfVectorizer()
    with pytest.raises(ValueError):
        tfidf.fit(dataset=tabular_dataset, field=Field("text", vocab=None))


@pytest.mark.usefixtures("tabular_dataset")
def test_transform_example_none(tabular_dataset):
    tabular_dataset.finalize_fields()
    text_field = tabular_dataset.field_dict["text"]
    vocab = text_field.vocab

    tfidf = TfIdfVectorizer(vocab=vocab)
    tfidf.fit(dataset=tabular_dataset, field=text_field)

    with pytest.raises(ValueError):
        tfidf.transform(examples=None)