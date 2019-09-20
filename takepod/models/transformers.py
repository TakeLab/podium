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


class FeatureTransformer:

    def __init__(self,
                 feature_extraction_fn: Callable[[NamedTuple], np.ndarray],
                 tensor_transformer: TensorTransformer):
        self.feature_extraction_fn = feature_extraction_fn
        self.tensor_transform = tensor_transformer

    def fit(self,
            x: NamedTuple,
            y: np.ndarray):
        x_tensor = self.feature_extraction_fn(x)
        self.tensor_transform.fit(x_tensor, y)

    def transform(self,
                  x: NamedTuple) -> np.ndarray:
        x_tensor = self.feature_extraction_fn(x)
        self.tensor_transform.transform(x_tensor)

