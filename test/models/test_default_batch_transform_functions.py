from collections import namedtuple

import pytest

from takepod.models import default_batch_transform


def test_default_batch_transform():
    x_batch_class = namedtuple("x_batch_class", ("x_field_1",))
    y_batch_class = namedtuple("y_batch_class", ("y_field_1",))

    x_batch = x_batch_class(x_field_1=1)
    y_batch = y_batch_class(y_field_1=2)

    x_value, y_value = default_batch_transform(x_batch, y_batch)

    assert x_value == 1
    assert y_value == 2


def test_default_batch_transform_raise_error():
    x_batch_class_1 = namedtuple("x_batch_class_1", ("x_field_1",))
    y_batch_class_1 = namedtuple("y_batch_class_1", ("y_field_1",))

    x_batch_class_2 = namedtuple("x_batch_class_2", ("x_field_1", "x_field_2"))
    y_batch_class_2 = namedtuple("y_batch_class_2", ("y_field_1", "y_field_2"))

    x_batch = x_batch_class_2(x_field_1=1, x_field_2=2)
    y_batch = y_batch_class_1(y_field_1=3)

    with pytest.raises(RuntimeError):
        default_batch_transform(x_batch, y_batch)

    x_batch = x_batch_class_1(x_field_1=1)
    y_batch = y_batch_class_2(y_field_1=2, y_field_2=3)

    with pytest.raises(RuntimeError):
        default_batch_transform(x_batch, y_batch)

    x_batch = x_batch_class_2(x_field_1=1, x_field_2=2)
    y_batch = y_batch_class_2(y_field_1=3, y_field_2=4)

    with pytest.raises(RuntimeError):
        default_batch_transform(x_batch, y_batch)
