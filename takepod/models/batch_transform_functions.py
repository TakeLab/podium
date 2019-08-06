from typing import NamedTuple, Tuple
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


def default_batch_transform(x_batch: NamedTuple,
                            y_batch: NamedTuple
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """The default batch transform function. If the `x_batch` and `y_batch` parameters
    contain only a a single attribute (field in the context of a Podium Example)
    each, that field's value is returned.

    Parameters
    ----------
    x_batch: NamedTuple
        The input features of the batch.

    y_batch: NamedTuple
        The labels of the current batch.

    Returns
    -------
    x, y
        Returns a tuple containing the only features and labels in the batch.

    Raises
    ------
    RuntimeError
        In case the input batch features and labels (`x_batch`, and `y_batch` parameters)
        don't have a single attribute (field) each.
    """
    x = default_feature_transform(x_batch)
    y = default_label_transform(y_batch)

    return x, y


def default_feature_transform(x_batch):
    if len(x_batch) != 1:
        error_msg = "x_batch length must be 1" \
                    " if the default batch transform function is used."
        _LOGGER.error(error_msg)
        raise RuntimeError(error_msg)

    return x_batch[0]


def default_label_transform(y_batch):
    if len(y_batch) != 1:
        error_msg = "y_batch length must be 1" \
                    " if the default batch transform function is used."
        _LOGGER.error(error_msg)
        raise RuntimeError(error_msg)

    return y_batch[0]
