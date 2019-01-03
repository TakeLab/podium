import pytest

from takepod.bayopt.evaluator import Evaluator


def evaluation_function(x, y, z):
    return x * y - z


ARGUMENT_DICTS = [
    {"x": 2, "y": 3, "z": 4},
    {"x": 3, "y": -3, "z": -2},
    {"x": 0, "y": 0, "z": 0},
    {"z": 2, "x": 3, "y": 1}
]

EXPECTED_RESULTS = [2, -7, 0, 1]


@pytest.mark.parametrize(
    "argument_dict, expected_result",
    list(zip(ARGUMENT_DICTS, EXPECTED_RESULTS))
)
def test_call_correct_evaluation(argument_dict, expected_result):
    evaluator = Evaluator(evaluation_function)

    assert evaluator(argument_dict) == expected_result


def test_call_correct_history():
    evaluator = Evaluator(evaluation_function)

    for argument_dict in ARGUMENT_DICTS:
        evaluator(argument_dict)

    dict_history_results = zip(
        ARGUMENT_DICTS,
        evaluator.history,
        EXPECTED_RESULTS
    )

    for expected_arg_dict, history, expected_res in dict_history_results:
        received_argument_dict, received_result = history

        assert expected_arg_dict == received_argument_dict
        assert expected_res == received_result
