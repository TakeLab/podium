import numpy as np

from takepod.bayopt.hyperparameter_definition import HyperparameterDefinition

HPARAM_DEFINITION_DICT = {
    "a": ("categorical", ("positive", "negative", "neutral")),
    "b": ("real", (1.5, 4.8)),
    "c": ("integer", (3, 8))
}

HPARAM_VALUES_DICT = {
    "a": "negative",
    "b": 3.2,
    "c": 6
}

# the real vector does NOT have components corresponding to categorical
# variables one-hot encoded and components corresponding to integer variables
# rounded
REAL_VECTOR = np.array([0.3, 0.82, 0.05, 3.2, 6.5])

MIXED_VECTOR = np.array([0.0, 1.0, 0.0, 3.2, 6.0])

BOUNDS = np.array(
    [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.5, 4.8], [3.0, 8.0]]
)


def test_correct_bounds():
    hparam_def = HyperparameterDefinition(HPARAM_DEFINITION_DICT)

    assert (BOUNDS == hparam_def.bounds).all()


def test_dict_to_real_vector():
    hparam_def = HyperparameterDefinition(HPARAM_DEFINITION_DICT)

    received_real_vector = hparam_def.dict_to_real_vector(HPARAM_VALUES_DICT)

    # when transforming from a dict, the resultant real vector is
    # actually rounded and one-hot encoded (mixed vector)
    expected_real_vector = MIXED_VECTOR

    assert (expected_real_vector == received_real_vector).all()


def test_real_vector_to_dict():
    hparam_def = HyperparameterDefinition(HPARAM_DEFINITION_DICT)

    received_dict = hparam_def.real_vector_to_dict(REAL_VECTOR)
    expected_dict = HPARAM_VALUES_DICT

    assert expected_dict == received_dict


def test_real_vector_to_mixed_vector():
    hparam_def = HyperparameterDefinition(HPARAM_DEFINITION_DICT)

    received_mixed_vector = hparam_def.real_vector_to_mixed_vector(REAL_VECTOR)
    expected_mixed_vector = MIXED_VECTOR

    assert (expected_mixed_vector == received_mixed_vector).all()
