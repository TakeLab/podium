"""Module contains interfaces for a trainer."""
from abc import ABC, abstractmethod


class AbstractTrainer(ABC):
    """Interface for base trainer that can train the model."""

    @abstractmethod
    def train(self, model, iterator, **kwargs):
        """Method trains a model with data from given iterator.

        Parameters
        ----------
        model : AbstractSupervisedModel
            model that needs to be trained
        iterator : Iterator
            iterator instance that provides data from a dataset
        kwargs : dict
            trainer specific parameters
        """
        pass
