"""Module contains simple trainer classes."""
from takepod.models.base_trainer import AbstractTrainer


class SimpleTrainer(AbstractTrainer):
    """Simple trainer"""

    MAX_EPOCH_KEY = "max_epoch"

    def __init__(self, model):
        """Constructor that initializes trainer.

        Parameters
        ----------
            model : AbstractSupervisedModel
        """
        super(SimpleTrainer, self).__init__(model=model)

    def train(self, iterator, **kwargs):
        for (x_batch, y_batch) in iterator:
            if iterator.epoch > kwargs[SimpleTrainer.MAX_EPOCH_KEY]:
                break
            x_train = x_batch[0]
            y_train = y_batch[0].ravel()
            self.model.fit(X=x_train, y=y_train)
