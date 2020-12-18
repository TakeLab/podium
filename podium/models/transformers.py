from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import numpy as np


class TensorTransformer(ABC):
    """
    Abstract class used to transform tensors.

    Used in feature pre-processing during training and prediction. Usually used
    in FeatureTransformer to transform tensors returned by the feature
    extraction callable.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the transformer to the provided data.

        Parameters
        ----------
        x : np.ndarray
            Features in numpy array form.
        y : np.ndarray
            Labels in numpy array form.
        """
        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms the passed features.

        Parameters
        ----------
        x: np.ndarray
            Features to be transformed in numpy array form.

        Returns
        -------
        np.array
            Transformed features.
        """
        pass

    @abstractmethod
    def requires_fitting(self) -> bool:
        """
        Returns True if this TensorTransformer requires fitting.

        Returns
        -------
        True if this TensorTransformer requires fitting, else returns False.
        """
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)


class SklearnTensorTransformerWrapper(TensorTransformer):
    """
    Wrapper class for Sklearn feature transformers.
    """

    def __init__(self, feature_transformer, requires_fitting: bool = True) -> None:
        """
        Creates a new SklearnTensorTransformerWrapper.

        Parameters
        ----------
        feature_transformer
            The sklearn feature transformer to be wrapped. Example of this would be
            a sklearn pipeline containing a sequence of transformations.

        requires_fitting: bool
            Whether this tensor transformer should be fitted.
        """
        self.feature_transformer = feature_transformer
        self.requires_fitting_flag = requires_fitting

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.requires_fitting():
            self.feature_transformer.fit(x, y)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.feature_transformer.transform(x)

    def requires_fitting(self) -> bool:
        return self.requires_fitting_flag


class FeatureTransformer:
    """
    Class used to transform Dataset batches into features used in model
    prediction and training.
    """

    def __init__(
        self,
        feature_extraction_fn: Callable[[NamedTuple], np.ndarray],
        tensor_transformer: TensorTransformer = None,
    ) -> None:
        """
        Creates a new FeatureTransformer.

        Parameters
        ----------
        feature_extraction_fn: Callable[[NamedTuple], np.ndarray]
            Callable that takes a podium feature batch as an argument and returns a
            numpy array representing the data.

        tensor_transformer: TensorTransformer
            TensorTransformer used to transform the transform the tensors provided by the
            `feature_extraction_fn` callable.
        """
        self.feature_extraction_fn = feature_extraction_fn
        self.tensor_transformer = tensor_transformer

    def fit(self, x: NamedTuple, y: np.ndarray) -> None:
        """
        Fits this tensor transformer to the provided data.

        Parameters
        ----------
        x: NamedTuple
            Podium feature batch containing the features to be transformed.

        y: np.ndarray
            Labels corresponding to the features in `x`.
        """
        if not self.requires_fitting():
            return

        x_tensor = self.feature_extraction_fn(x)
        self.tensor_transformer.fit(x_tensor, y)

    def transform(self, x: NamedTuple) -> np.ndarray:
        """
        Trasforms the provided podium feature batch into a numpy array.

        Parameters
        ----------
        x: NamedTuple
            Feature batch to be transformed.

        Returns
        -------
        np.ndarray
            Transformed features.
        """
        x_tensor = self.feature_extraction_fn(x)
        if self.tensor_transformer is None:
            return x_tensor

        else:
            return self.tensor_transformer.transform(x_tensor)

    def __call__(self, x: NamedTuple) -> np.ndarray:
        """Trasforms the provided podium feature batch into a numpy array.
        Parameters
        ----------
        x: NamedTuple
            Feature batch to be transformed.

        Returns
        -------
        np.ndarray
            Transformed features.
        """
        return self.transform(x)

    def requires_fitting(self) -> bool:
        """
        Returns True if the contained TensorTransformer exists and requires
        fitting, else returns None.

        Returns
        -------
        bool
            True if the contained TensorTransformer exists and requires fitting,
            else returns False.
        """
        return (
            self.tensor_transformer is not None
            and self.tensor_transformer.requires_fitting()
        )
