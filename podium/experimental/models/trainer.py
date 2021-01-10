"""
Module contains interfaces for a trainer.
"""
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import numpy as np

from podium.datasets import Dataset

from .model import AbstractSupervisedModel
from .transformers import FeatureTransformer


class AbstractTrainer(ABC):
    """
    Interface for base trainer that can train the model.
    """

    @abstractmethod
    def train(
        self,
        model: AbstractSupervisedModel,
        dataset: Dataset,
        feature_transformer: FeatureTransformer,
        label_transform_fun: Callable[[NamedTuple], np.ndarray],
        **kwargs,
    ):

        """
        Method trains a model with data from given Iterator.

        Parameters
        ----------
        model : AbstractSupervisedModel
            The model that needs to be trained.
        dataset : Dataset
            Dataset the model will be trained on
        feature_transformer: Callable[[NamedTuple], np.ndarray]
            Callable that transforms the input part of the batch returned by the iterator
            into features that can be fed into the model.
        label_transform_fun: Callable[[NamedTuple], np.ndarray]
            Callable that transforms the target part of the batch returned by the iterator
            into the same format the model prediction is. For a hypothetical perfect model
            the prediction result of the model for some examples must be identical to the
            result of this callable for those same examples.
        kwargs : dict
            Trainer specific parameters.
        """

        pass
