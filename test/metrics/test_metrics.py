import pytest
from takepod.metrics.metrics import f1_metric


def test_f1():
    true = [1, 1]
    pred = [0, 1]
    assert pytest.approx(f1_metric(true, pred), 0.001) == 0.666
