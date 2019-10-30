from collections import namedtuple
import numpy as np
import pytest
import takepod.examples.model_example as example


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
    X = example.feature_extraction_fn(input_batch)
    y = example.label_extraction_fun(target_batch)
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
