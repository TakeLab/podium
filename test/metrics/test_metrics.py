from takepod.metrics import standard_metrics
import pytest


def test_f1():
    true = [1, 1]
    pred = [0, 1]
    assert 0.666 == pytest.approx(standard_metrics.f1(true, pred), 0.001)
