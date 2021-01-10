"""
Module contains base model interfaces.
"""
from abc import ABC, abstractmethod


class AbstractSupervisedModel(ABC):
    """
    Interface for supervised models.

    Attributes
    ----------
    PREDICTION_KEY : str
        key for defining prediction return variable
    """

    PREDICTION_KEY = "pred"

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Method trains the model and returns dictionary of values defined by
        model specific key parameters.

        Parameters
        ----------
        X : np.array
            input data
        y : np.array
            data labels
        **kwargs : dict
            Additional key-value parameters for model

        Returns
        -------
        result : dict
            dictionary mapping fit results to defined model specific key
            parameters
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Predict labels for given data.

        Parameters
        ----------
        X : np.array
            input data
        **kwargs : dict
            Additional key-value parameters for model

        Returns
        -------
        result : dict
            dictionary mapping fit results to defined model specific key
            parameters
        """
        pass

    @abstractmethod
    def reset(self, **kwargs):
        """
        Resets the model to its initial state so it can be re-trained.

        Parameters
        ----------
        kwargs
            Additional key-value parameters for model
        """
        pass


class AbstractFrameworkModel(ABC):
    """
    Interface for framework models.
    """

    @abstractmethod
    def save(self, file_path, **kwargs):
        """
        Method saves model to given file_path with additional arguments defined
        in kwargs.

        Parameters
        ----------
        file_path : str
            path to file where the model should be saved
        **kwargs : dict
            Additional key-value parameters for saving mode

        Raises
        ------
        IOError
            if there was an error while writing to a file
        """
        pass

    @abstractmethod
    def load(self, **kwargs):
        """
        Method loads model from given file_path with additional arguments
        defined in kwargs.

        Parameters
        ----------
        file_path : str
            path to file where the model should be saved
        **kwargs : dict
            Additional key-value parameters for loading model

        Returns
        -------
        model
            method returns loaded model

        Raises
        ------
        ValueError
            if the given path doesn't exist
        IOError
            if there was an error while reading from a file
        """
        pass
