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

    def train(self,
              model,
              iterator,
              feature_transform_fun,
              label_transform_fun,
              **kwargs):
        self._check_kwargs(**kwargs)
        for _ in range(kwargs[SimpleTrainer.MAX_EPOCH_KEY]):
            for x_batch, y_batch in iterator:
                x = feature_transform_fun(x_batch)
                y = label_transform_fun(y_batch)
                model.fit(X=x, y=y)

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
