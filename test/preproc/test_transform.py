from takepod.preproc.transform import (
    make_bow_vector,
    create_word_to_index,
    categories_to_int
)


def test_create_word_to_index():
    docs = [
        'Brown fox jumps',
        'Red riding hood scares big bad wolf'
    ]
    word_to_ix = create_word_to_index(docs)
    assert word_to_ix == {
        "Brown": 0,
        "fox": 1,
        "jumps": 2,
        "Red": 3,
        "riding": 4,
        "hood": 5,
        "scares": 6,
        "big": 7,
        "bad": 8,
        "wolf": 9
    }


def test_create_word_to_index_duplicate_words():
    docs = [
        'Brown fox jumps',
        'brown fox'
    ]
    word_to_ix = create_word_to_index(docs)
    assert word_to_ix == {
        "Brown": 0,
        "fox": 1,
        "jumps": 2,
        "brown": 3
    }


def test_create_bow_vector():
    docs = [
        'Brown fox jumps',
        'brown fox'
    ]
    word_to_ix = create_word_to_index(docs)
    document_to_transform = 'Brown fox'
    transformed = make_bow_vector(document_to_transform, word_to_ix)
    # tensor dimension is of word_to_ix size
    assert transformed == [1, 1, 0, 0]


def test_categories_to_int():
    labels = ['Negative', 'Positive', 'Negative']
    assert categories_to_int(labels) == [0, 1, 0]


def test_categories_to_int_custom_mapping():
    labels = ['Positive', 'Negative', 'Positive']
    mapping = {'Positive': 1, 'Negative': 0}
    assert categories_to_int(labels, mapping) == [1, 0, 1]
