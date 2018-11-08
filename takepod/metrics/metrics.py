"""Module contains custom metrics."""
from sklearn.metrics import f1_score


def f1_metric(true, pred):
    """Function calculates F1 score."""
    return f1_score(true, pred)
