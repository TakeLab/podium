from takepod.metrics.metrics import f1_metric
import pytest


def test_f1():
    true = [1, 1]
    pred = [0, 1]
    assert 0.666 == pytest.approx(f1_metric(true, pred), 0.001)
