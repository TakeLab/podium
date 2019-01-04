"""Module contains interfaces for a trainer."""
from abc import ABC, abstractmethod


class AbstractTrainer(ABC):
    """Interface for base trainer that can train the model."""

    @abstractmethod
    def train(self, iterator, **kwargs):
        """Method trains a model with data from given iterator.

        Parameters
        ----------
        iterator : Iterator
            iterator instance that provides data from a dataset
        kwargs : dict
            trainer specific parameters
        """
        pass
