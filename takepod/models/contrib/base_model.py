""""Module defines base models."""
from abc import ABC, abstractmethod


class ExportableModel(ABC):
    """Interface for models that can export parameters."""

    @abstractmethod
    def export(self, weights):
        """Method exports parameters given by weights.

        Parameters
        ----------
        weights : iter
            params identificators
        """
        pass


class SupervisedModel(ABC):
    """Interface for supervised models."""
    @abstractmethod
    def train(self, data, labels, **kwargs):
        """Method trains the model

        Parameters
        ----------
        data : iter
            Iterable of unpreprocesed input data
        labels : iter
            Iterable of unpreprocessed labels
        **kwargs : dict
            Additional key-value parameters for model

        Returns
        -------
        total_loss : tensor
            loss for the final training epoch
        """
        pass

    @abstractmethod
    def test(self, data, **kwargs):
        """Predict labels for given data

        Parameters
        ----------
        data : iter
            iter of unpreprocesed input data
        **kwargs : dict
            Additional key-value parameters for model

        Returns
        -------
        predicted : tensor
            Predicted output labels
        """
        pass
