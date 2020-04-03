"""Module contains functions used to transform batch to tensors that models accept."""
import logging

_LOGGER = logging.getLogger(__name__)


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
