import pytest


datasets = pytest.importorskip("datasets")
from datasets import ClassLabel, Features, Translation

from podium.dataload.huggingface_dataset_converter import (
    HuggingFaceDatasetConverter,
    convert_features_to_fields,
)
from podium.datasets import Dataset


SIMPLE_DATA = {
    "id": [None, 1],
    "name": ["Se7en", "Ford v Ferrari"],
    "review": ["Pretty good", "Could be better"],
    "rating": [7.2, 4.2],
    "related_movies": [[2, 4], [5, 20]],
}

COMPLEX_DATA = {
    "translation": [
        {"en": "this is bad", "fr": "c'est mauvais"},
        {"en": "this is good", "fr": "c'est bon"},
    ],
    "sentiment": [0, 1],
}


@pytest.fixture
def simple_dataset():
    return datasets.Dataset.from_dict(SIMPLE_DATA)


@pytest.fixture
def complex_dataset():
    features = {
        "translation": Translation(languages=("en", "fr")),
        "sentiment": ClassLabel(num_classes=2),
    }

    return datasets.Dataset.from_dict(COMPLEX_DATA, Features(features))


@pytest.fixture
def dataset_dict(simple_dataset, complex_dataset):
    return {"simple": simple_dataset, "complex": complex_dataset}


def test_simple_feature_conversion(simple_dataset):
    fields = convert_features_to_fields(simple_dataset.features)

    assert not fields["id"]._keep_raw
    assert not fields["rating"]._keep_raw
    assert not fields["related_movies"]._keep_raw

    assert fields["id"]._allow_missing_data
    assert fields["name"]._allow_missing_data
    assert fields["review"]._allow_missing_data
    assert fields["rating"]._allow_missing_data
    assert fields["related_movies"]._allow_missing_data


def test_simple_data(simple_dataset):
    converter_iter = iter(HuggingFaceDatasetConverter(simple_dataset))

    example1 = next(converter_iter)
    assert example1["id"][1] == SIMPLE_DATA["id"][0]
    assert example1["name"][1] == SIMPLE_DATA["name"][0].split()
    assert example1["review"][1] == SIMPLE_DATA["review"][0].split()
    assert example1["rating"][1] == SIMPLE_DATA["rating"][0]
    assert example1["related_movies"][1] == SIMPLE_DATA["related_movies"][0]

    example2 = next(converter_iter)
    assert example2["id"][1] == SIMPLE_DATA["id"][1]
    assert example2["name"][1] == SIMPLE_DATA["name"][1].split()
    assert example2["review"][1] == SIMPLE_DATA["review"][1].split()
    assert example2["rating"][1] == SIMPLE_DATA["rating"][1]
    assert example2["related_movies"][1] == SIMPLE_DATA["related_movies"][1]


def test_complex_feature_conversion(complex_dataset):
    fields = convert_features_to_fields(complex_dataset.features)

    assert not fields["translation"]._keep_raw
    assert fields["sentiment"].is_target

    assert fields["translation"]._allow_missing_data
    assert not fields["sentiment"]._allow_missing_data


def test_complex_data(complex_dataset):
    converter_iter = iter(HuggingFaceDatasetConverter(complex_dataset))

    example1 = next(converter_iter)
    assert example1["translation"][1] == COMPLEX_DATA["translation"][0]
    assert example1["sentiment"][1] == COMPLEX_DATA["sentiment"][0]

    example2 = next(converter_iter)
    assert example2["translation"][1] == COMPLEX_DATA["translation"][1]
    assert example2["sentiment"][1] == COMPLEX_DATA["sentiment"][1]


def test_invalid_dataset():
    with pytest.raises(TypeError):
        HuggingFaceDatasetConverter({"data": [1, 2, 3]})


def test_as_dataset(simple_dataset):
    podium_dataset = HuggingFaceDatasetConverter(simple_dataset).as_dataset()
    assert isinstance(podium_dataset, Dataset)
    assert len(podium_dataset) == 2


def test_from_dataset_dict(dataset_dict):
    dataset_dict_converted = HuggingFaceDatasetConverter.from_dataset_dict(dataset_dict)

    assert isinstance(dataset_dict_converted, dict)
    assert isinstance(
        dataset_dict_converted["simple"], HuggingFaceDatasetConverter
    ) and isinstance(dataset_dict_converted["complex"], HuggingFaceDatasetConverter)
