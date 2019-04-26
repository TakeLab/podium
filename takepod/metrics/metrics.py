"""Module contains custom metrics."""
import logging

_LOGGER = logging.getLogger(__name__)

try:
    from sklearn.metrics import f1_score
except ImportError as ex:
    _LOGGER.debug("Problem occured while trying to import sklearn. If the "
                  "library is not installed visit https://scikit-learn.org"
                  " for more details.")


def f1_metric(true, pred):
    """Function calculates F1 score."""
    return f1_score(true, pred)
