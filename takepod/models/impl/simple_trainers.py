"""Module contains simple trainer classes."""
from takepod.models.trainer import AbstractTrainer


class SimpleTrainer(AbstractTrainer):
    """Simple trainer class.

    Attributes
    ----------
    MAX_EPOCH_KEY : str
        keyword argument key for maximal number of epochs used for training
    """

    MAX_EPOCH_KEY = "max_epoch"

    def train(self, model, iterator, batch_transform=None, **kwargs):
        self._check_kwargs(**kwargs)
        for _ in range(kwargs[SimpleTrainer.MAX_EPOCH_KEY]):
            for x_batch, y_batch in iterator:
                if batch_transform is not None:
                    x_batch, y_batch = batch_transform(x_batch, y_batch)
                model.fit(X=x_batch, y=y_batch)

    def _check_kwargs(self, **kwargs):
        """Method checks if kwargs contains necessary training parameters.

        Parameters
        ----------
        kwargs : dict
            training parameters
        """
        if self.MAX_EPOCH_KEY not in kwargs:
            raise ValueError("Missing training parameter: MAX_EPOCH "
                             "(used for determining stop criterion)")
