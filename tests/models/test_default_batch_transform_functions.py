from collections import namedtuple

import pytest

from podium.models.batch_transform_functions import default_label_transform, \
    default_feature_transform


def test_default_batch_transform_functions():
    x_batch_class = namedtuple("x_batch_class", ("x_field_1",))
    y_batch_class = namedtuple("y_batch_class", ("y_field_1",))

    x_batch = x_batch_class(x_field_1=1)
    y_batch = y_batch_class(y_field_1=2)

    x_value = default_feature_transform(x_batch)
    assert x_value == 1

    y_value = default_label_transform(y_batch)
    assert y_value == 2


def test_default_batch_transform_raise_error():

    x_batch_class_2 = namedtuple("x_batch_class_2", ("x_field_1", "x_field_2"))
    y_batch_class_2 = namedtuple("y_batch_class_2", ("y_field_1", "y_field_2"))

    x_batch = x_batch_class_2(x_field_1=1, x_field_2=2)
    y_batch = y_batch_class_2(y_field_1=2, y_field_2=3)

    with pytest.raises(RuntimeError):
        default_feature_transform(x_batch)

    with pytest.raises(RuntimeError):
        default_label_transform(y_batch)
