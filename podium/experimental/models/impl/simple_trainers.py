"""
Module contains simple trainer classes.
"""
from podium.datasets import Iterator
from podium.experimental.models.trainer import AbstractTrainer


class SimpleTrainer(AbstractTrainer):
    """
    Simple trainer class.

    Attributes
    ----------
    MAX_EPOCH_KEY : str
        keyword argument key for maximal number of epochs used for training
    """

    MAX_EPOCH_KEY = "max_epoch"

    def train(
        self,
        model,
        dataset,
        feature_transformer,
        label_transform_fun,
        max_epoch,
        iterator=None,
    ):

        if iterator is None:
            iterator = Iterator()

        for _ in range(max_epoch):
            for batch in iterator(dataset):
                x = feature_transformer.transform(batch)
                y = label_transform_fun(batch)
                model.fit(X=x, y=y)

    def _check_kwargs(self, **kwargs):
        """
        Method checks if kwargs contains necessary training parameters.

        Parameters
        ----------
        kwargs : dict
            training parameters
        """
        if self.MAX_EPOCH_KEY not in kwargs:
            raise ValueError(
                f"Missing training parameter: {self.MAX_EPOCH_KEY} "
                "(used for determining stop criterion)"
            )
