import pytest

pytest.importorskip('datasets')
from datasets import ClassLabel, Dataset, Features, Translation

from podium.dataload.huggingface_dataset_converter import (convert_features_to_fields,
                                                           HuggingFaceDatasetConverter)


SIMPLE_DATA = {
    'id': [None, 1],
    'name': ['Se7en', 'Ford v Ferrari'],
    'review': ['Pretty good', 'Could be better'],
    'rating': [7.2, 4.2],
    'related_movies': [[2, 4], [5, 20]],
}

COMPLEX_DATA = {
    'translation': [
        {'en': 'this is bad', 'fr': "c'est mauvais"},
        {'en': 'this is good', 'fr': "c'est bon"}
    ],
    'sentiment': [0, 1]
}


@pytest.fixture
def simple_dataset():
    return Dataset.from_dict(SIMPLE_DATA)


@pytest.fixture
def complex_dataset():
    features = {
        'translation': Translation(languages=('en', 'fr')),
        'sentiment': ClassLabel(num_classes=2),
    }

    return Dataset.from_dict(COMPLEX_DATA, Features(features))


def test_simple_feature_conversion(simple_dataset):
    fields = convert_features_to_fields(simple_dataset.features)

    assert fields['id'].store_as_raw
    assert fields['name'].is_sequential
    assert fields['review'].is_sequential
    assert fields['rating'].store_as_raw
    assert fields['related_movies'].store_as_raw


def test_simple_data(simple_dataset):
    converter_iter = iter(HuggingFaceDatasetConverter(simple_dataset))

    example1 = next(converter_iter)
    assert example1.id[0] == SIMPLE_DATA['id'][0]
    assert example1.name[1] == SIMPLE_DATA['name'][0].split()
    assert example1.review[1] == SIMPLE_DATA['review'][0].split()
    assert example1.rating[0] == SIMPLE_DATA['rating'][0]
    assert example1.related_movies[0] == SIMPLE_DATA['related_movies'][0]

    example2 = next(converter_iter)
    assert example2.id[0] == SIMPLE_DATA['id'][1]
    assert example2.name[1] == SIMPLE_DATA['name'][1].split()
    assert example2.review[1] == SIMPLE_DATA['review'][1].split()
    assert example2.rating[0] == SIMPLE_DATA['rating'][1]
    assert example2.related_movies[0] == SIMPLE_DATA['related_movies'][1]


def test_complex_feature_conversion(complex_dataset):
    fields = convert_features_to_fields(complex_dataset.features)

    assert fields['translation'].store_as_raw
    assert fields['sentiment'].is_target


def test_complex_data(complex_dataset):
    converter_iter = iter(HuggingFaceDatasetConverter(complex_dataset))

    example1 = next(converter_iter)
    assert example1.translation[0] == COMPLEX_DATA['translation'][0]
    assert example1.sentiment[0] == COMPLEX_DATA['sentiment'][0]

    example2 = next(converter_iter)
    assert example2.translation[0] == COMPLEX_DATA['translation'][1]
    assert example2.sentiment[0] == COMPLEX_DATA['sentiment'][1]


def test_as_dataset(simple_dataset):
    assert len(HuggingFaceDatasetConverter(simple_dataset).as_dataset()) == 2
