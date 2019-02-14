from collections import namedtuple
import numpy as np
import pytest
import takepod.examples.model_example as example
from takepod.datasets.pauza_dataset import PauzaHRDataset
from takepod.models.simple_trainers import SimpleTrainer
from takepod.models.fc_model import ScikitMLPClassifier
from takepod.storage.iterator import Iterator


def test_basic_pauza_hr_fields():
    fields = example.basic_pauza_hr_fields()
    assert len(fields) == 2
    assert "Text" in fields
    assert fields["Text"].name == "Text"
    assert "Rating" in fields
    assert fields["Rating"].name == "Rating"


def test_basic_batch_transform_fun():
    input_class = namedtuple("InputBatch", ["Text"])
    target_class = namedtuple("TargetBatch", ["Rating"])
    input_batch = input_class(**{"Text": np.array([[1, 3], [2, 3], [0, 4]])})
    target_batch = target_class(**{"Rating": np.array([1, 0, 4])})
    X, y = example.basic_batch_transform_fun(input_batch, target_batch)
    assert X.shape == (3, 2)
    assert y.shape == (3,)


@pytest.mark.parametrize(
    "rating, expected_output",
    [
        (0, 0),
        (0.5, 1),
        (3, 6),
        (5.5, 11)
    ]
)
def test_numericalize_pauza_rating(rating, expected_output):
    assert example.numericalize_pauza_rating(rating) == expected_output


def test_batch_transform_fun_vectorize_avg():
    embedding_matrix = np.array([[0, 1, 1, 4, 4],
                                 [9, 7, 4, 3, 6],
                                 [5, 3, 1, 2, 7],
                                 [4, 7, 9, 3, 4],
                                 [5, 6, 3, 1, 0]])
    input_class = namedtuple(
        "InputBatch", ["Text"])
    target_class = namedtuple(
        "TargetBatch", ["Rating"])
    input_batch = input_class(
        **{"Text": np.array([[1, 3], [2, 3], [0, 4]])})
    target_batch = target_class(
        **{"Rating": np.array([1, 0, 4])})

    X, y = example.batch_transform_fun_vectorize_avg(
        x_batch=input_batch,
        y_batch=target_batch,
        embedding_matrix=embedding_matrix)
    assert X.shape == (3, 5)
    assert y.shape == (3,)
