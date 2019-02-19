"""Module contains simple trainer classes."""
from takepod.models.base_trainer import AbstractTrainer


class SimpleTrainer(AbstractTrainer):
    """Simple trainer class.

    Attributes
    ----------
    MAX_EPOCH_KEY : str
        keyword argument key for maximal number of epochs used for training
    BATCH_TRANSFORM_FUN_KEY : str
        keyword arugment key for function that can transform batch for a model
    """

    MAX_EPOCH_KEY = "max_epoch"
    BATCH_TRANSFORM_FUN_KEY = "batch_transform"

    def __init__(self, model):
        """Constructor that initializes trainer.

        Parameters
        ----------
            model : AbstractSupervisedModel
        """
        super(SimpleTrainer, self).__init__(model=model)

    def train(self, iterator, **kwargs):
        self._check_kwargs(**kwargs)
        batch_transform_fun = kwargs.get(
            SimpleTrainer.BATCH_TRANSFORM_FUN_KEY, None)
        for _ in range(kwargs[SimpleTrainer.MAX_EPOCH_KEY]):
            for x_batch, y_batch in iterator:
                if batch_transform_fun:
                    x_batch, y_batch = batch_transform_fun(x_batch, y_batch)
                self.model.fit(X=x_batch, y=y_batch)

    def _check_kwargs(self, **kwargs):
        """Method checks if kwargs contains neccessary training parameters.

        Parameters
        ----------
        kwargs : dict
            training parameters
        """
        if self.MAX_EPOCH_KEY not in kwargs:
            raise ValueError("Missing training parameter: MAX_EPOCH "
                             "(used for determining stop criterion)")
