"""
Module contains functions used to transform batch to tensors that models accept.
"""


def default_feature_transform(x_batch):
    if len(x_batch) != 1:
        raise RuntimeError(
            "x_batch length must be 1 if the default batch transform function is used."
        )

    return x_batch[0]


def default_label_transform(y_batch):
    if len(y_batch) != 1:
        raise RuntimeError(
            "y_batch length must be 1 if the default batch transform function is used."
        )

    return y_batch[0]
