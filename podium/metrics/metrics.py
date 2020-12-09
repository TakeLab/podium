"""Module contains custom metrics."""
try:
    from sklearn.metrics import f1_score
except ImportError:
    print(
        "Problem occured while trying to import sklearn. If the "
        "library is not installed visit https://scikit-learn.org"
        " for more details."
    )
    raise


def f1_metric(true, pred):
    """Function calculates F1 score."""
    return f1_score(true, pred)


def multiclass_f1_metric(true, pred, average="weighted"):
    """Function calculates F1 score on multiclass classification."""
    return f1_score(true, pred, average=average)
