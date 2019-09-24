from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import numpy as np


class TensorTransformer(ABC):

    @abstractmethod
    def fit(self,
            x: np.ndarray,
            y: np.ndarray):
        pass

    @abstractmethod
    def transform(self,
                  x: np.array
                  ) -> np.ndarray:
        pass


class DummyTensorTransformer(TensorTransformer):

    def fit(self,
            x: np.ndarray,
            y: np.ndarray):
        pass

    def transform(self,
                  x: np.array
                  ) -> np.ndarray:
        return x


# TODO add mechanism for Feature transformer to know if its tensor_transformer needs
#  fitting so batching can be avoided by callers.
class FeatureTransformer:

    def __init__(self,
                 feature_extraction_fn: Callable[[NamedTuple], np.ndarray],
                 tensor_transformer: TensorTransformer = None,
                 requires_fitting=True):
        self.feature_extraction_fn = feature_extraction_fn
        self.tensor_transformer = tensor_transformer
        self.requires_fitting_flag = requires_fitting

    def fit(self,
            x: NamedTuple,
            y: np.ndarray):
        if not self.requires_fitting():
            return

        x_tensor = self.feature_extraction_fn(x)
        self.tensor_transformer.fit(x_tensor, y)

    def transform(self,
                  x: NamedTuple) -> np.ndarray:
        x_tensor = self.feature_extraction_fn(x)
        if self.tensor_transformer is None:
            return x_tensor

        else:
            self.tensor_transformer.transform(x_tensor)

    def requires_fitting(self):
        return self.tensor_transformer is not None and self.requires_fitting_flag
