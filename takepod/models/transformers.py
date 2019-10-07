from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import numpy as np


class TensorTransformer(ABC):
    """Abstract class used to transform tensors. Used in feature pre-processing during
    training and prediction.
    """

    @abstractmethod
    def fit(self,
            x: np.ndarray,
            y: np.ndarray):
        """Fits the transformer to the provided data.

        Parameters
        ----------

        x: np.ndarray
            Features in numpy array form.
        y: np.ndarray
            Labels in numpy array form.
        """
        pass

    @abstractmethod
    def transform(self,
                  x: np.array
                  ) -> np.ndarray:
        """Transforms the passed features.

        Parameters
        ----------
        x: np.ndarray
            Features to be transformed in numpy array form.

        Returns
        -------
        np.array
            Transformed features."""
        pass

    def requires_fitting(self) -> bool:
        return True


class SklearnTensorTransformerWrapper(TensorTransformer):

    def __init__(self,
                 feature_transformer):
        self.feature_transformer = feature_transformer

    def fit(self,
            x: np.ndarray,
            y: np.ndarray):
        self.feature_transformer.fit(x, y)

    def transform(self,
                  x: np.array
                  ) -> np.ndarray:
        return self.feature_transformer.transform(x)


# TODO add mechanism for Feature transformer to know if its tensor_transformer needs
#  fitting so batching can be avoided by callers.
class FeatureTransformer:
    """Class used to transform podium batches into features used in model prediction and
    training."""

    def __init__(self,
                 feature_extraction_fn: Callable[[NamedTuple], np.ndarray],
                 tensor_transformer: TensorTransformer = None):
        """Creates a new FeatureTransformer.

        Parameters
        ----------
        feature_extraction_fn: Callable[[NamedTuple], np.ndarray]
            Callable that takes a podium feature batch as an argument and returns a
            numpy tensor representing the data.
        tensor_transformer: TensorTransformer
            TensorTransformer used to transform the transform the tensors provided by the
            `feature_extraction_fn` callable.
        """
        self.feature_extraction_fn = feature_extraction_fn
        self.tensor_transformer = tensor_transformer

    def fit(self,
            x: NamedTuple,
            y: np.ndarray):
        """Fits this tensor transformer to the provided data.

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

    def transform(self,
                  x: NamedTuple) -> np.ndarray:
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

    def requires_fitting(self):
        return self.tensor_transformer is not None \
               and self.tensor_transformer.requires_fitting()
