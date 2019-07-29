"""Module contains interfaces for a trainer."""
from abc import ABC, abstractmethod


class AbstractTrainer(ABC):
    """Interface for base trainer that can train the model."""

    @abstractmethod
    def train(self, model, iterator, batch_transform=None, **kwargs):
        """Method trains a model with data from given iterator.

        Parameters
        ----------
        model : AbstractSupervisedModel
            model that needs to be trained
        iterator : Iterator
            iterator instance that provides data from a dataset
        batch_transform: callable(batch)
            function that transforms the batch returned by the iterator into
            a format that the model can accept
        kwargs : dict
            trainer specific parameters
        """
        pass
