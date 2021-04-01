"""Modules defines an experiment - class used to combine iteration over data,
model training and prediction."""
from inspect import isclass
from typing import Callable, Dict, NamedTuple, Type, Union

import numpy as np

from podium.datasets.dataset import Dataset
from podium.datasets.iterator import Iterator, SingleBatchIterator
from podium.utils.general_utils import repr_type_and_attrs

from .batch_transform_functions import default_feature_transform, default_label_transform
from .model import AbstractSupervisedModel
from .trainer import AbstractTrainer
from .transformers import FeatureTransformer


class Experiment:
    """
    Class used to streamline model fitting and prediction.
    """

    def __init__(
        self,
        model: Union[Type[AbstractSupervisedModel], AbstractSupervisedModel],
        trainer: AbstractTrainer = None,
        feature_transformer: Union[
            FeatureTransformer, Callable[[NamedTuple], np.array]
        ] = None,
        label_transform_fn: Callable[[NamedTuple], np.ndarray] = None,
    ):
        """
        Creates a new Experiment. The Experiment class is used to simplify model
        fitting and prediction using Podium components.

        Parameters
        ----------
        model : class or model instance
            Class of the Model to be fitted or a pre-trained model.
            If pre-trained model is passed and `fit` is called a new model instance will
            be created. For fine-tuning of the passed model instance call
            `partial_fit`.
            Must be a subclass of Podium's `AbstractSupervisedModel`

        trainer : AbstractTrainer
            Trainer used to fit the model.

        feature_transformer : Union[FeatureTransformer, Callable[[NamedTuple], np.array]
            FeatureTransformer that transforms the input part of the batch returned by the
            iterator into features that can be fed into the model. Will also be fitted
            during Experiment fitting.
            A callable taking an input batch and returning a numpy array of features can
            also be passed.
            If None, a default feature transformer that returns a single feature from
            the batch will be used. In this case the Dataset used in training must contain
            a single input field.

        label_transform_fn : Callable[[NamedTuple], np.ndarray]
            Callable that transforms the target part of the batch returned by the iterator
            into the same format the model prediction is. For a hypothetical perfect model
            the prediction result of the model for some examples must be identical to the
            result of this callable for those same examples.
            If None, a default label transformer that returns a single feature from
            the batch will be used. In this case the Dataset used in training must contain
            a single target field.
        """
        if isclass(model):
            self.model_class = model
            self.model = None
        else:
            self.model_class = model.__class__
            self.model = model

        self.trainer = trainer

        self.set_default_model_args()
        self.set_default_trainer_args()

        self.set_feature_transformer(feature_transformer)
        self.set_label_transformer(label_transform_fn)

    def set_default_model_args(self, **kwargs):
        """
        Sets the default model arguments. Model arguments are keyword arguments
        passed to the model constructor. Default arguments can be
        updated/overridden by arguments in the `model_kwargs` dict in the fit
        method.

        Parameters
        ----------
        kwargs
            Default model arguments.
        """
        self.default_model_args = kwargs

    def set_default_trainer_args(self, **kwargs):
        """Sets the default trainer arguments. Trainer arguments are keyword arguments
        passed to the trainer during model fitting. Default arguments can be
        updated/overridden by arguments in the `trainer_kwargs` parameter
        in the `fit` method.
        Parameters
        ----------
        kwargs
            Default trainer arguments.
        """
        self.default_trainer_args = kwargs

    def set_feature_transformer(self, feature_transformer):
        if feature_transformer is None:
            self.feature_transformer = FeatureTransformer(default_feature_transform)

        elif isinstance(feature_transformer, FeatureTransformer):
            self.feature_transformer = feature_transformer

        elif callable(feature_transformer):
            self.feature_transformer = FeatureTransformer(feature_transformer)

        else:
            raise TypeError(
                "Invalid feature_transformer. "
                "feature_transformer must be either "
                "be None, a FeatureTransformer instance or a callable "
                "taking a batch and returning a numpy matrix of features."
            )

    def set_label_transformer(self, label_transform_fn):
        self.label_transform_fn = (
            label_transform_fn
            if label_transform_fn is not None
            else default_label_transform
        )

    def fit(
        self,
        dataset: Dataset,
        model_kwargs: Dict = None,
        trainer_kwargs: Dict = None,
        feature_transformer: FeatureTransformer = None,
        trainer: AbstractTrainer = None,
    ):
        """
        Fits the model to the provided Dataset. During fitting, the provided
        Iterator and Trainer are used.

        Parameters
        ----------
        dataset : Dataset
            Dataset to fit the model to.

        model_kwargs : dict
            Dict containing model arguments. Arguments passed to the model are the default
            arguments defined with `set_default_model_args` updated/overridden by
            model_kwargs.

        trainer_kwargs : dict
            Dict containing trainer arguments. Arguments passed to the trainer are the
            default arguments defined with `set_default_trainer_args` updated/overridden
            by 'trainer_kwargs'.

        feature_transformer : FeatureTransformer, Optional
            FeatureTransformer that transforms the input part of the batch returned by the
            iterator into features that can be fed into the model. Will also be fitted
            during Experiment fitting.
            If None, the default FeatureTransformer provided in the constructor will be
            used. Otherwise, this will overwrite the default feature transformer.

        trainer : AbstractTrainer, Optional
            Trainer used to fit the model. If None, the trainer provided in the
            constructor will be used.

        training_iterator_callable: Callable[[Dataset], Iterator]
            Callable used to instantiate new instances of the Iterator used in fitting the
            model. If None, the training_iterator_callable provided in the
            constructor will be used.

        Raises
        ------
        RuntimeError
            If trainer is not provided either in the constructor or as an argument
            to the method.
        """

        model_kwargs = {} if model_kwargs is None else model_kwargs

        model_args = self.default_model_args.copy()
        model_args.update(model_kwargs)

        trainer = trainer if trainer is not None else self.trainer
        if trainer is None:
            raise RuntimeError(
                "No trainer provided. Trainer must be provided either in the "
                "constructor or as an argument to the fit method."
            )

        if feature_transformer is not None:
            self.set_feature_transformer(feature_transformer)

        # Fit the feature transformer if it needs fitting
        if self.feature_transformer.requires_fitting():
            for batch in SingleBatchIterator(dataset, shuffle=False):
                y = self.label_transform_fn(batch)
                self.feature_transformer.fit(batch, y)

        # Create new model instance
        self.model = self.model_class(**model_args)

        # Train the model
        self.partial_fit(dataset, trainer_kwargs, trainer)

    def partial_fit(
        self,
        dataset: Dataset,
        trainer_kwargs: Dict = None,
        trainer: AbstractTrainer = None,
    ):
        """
        Fits the model to the data without resetting the model.

        Parameters
        ----------
        dataset : Dataset
            Dataset to fit the model to.

        trainer_kwargs : dict
            Dict containing trainer arguments. Arguments passed to the trainer are the
            default arguments defined with `set_default_trainer_args` updated/overridden
            by 'trainer_kwargs'.

        trainer : AbstractTrainer, Optional
            Trainer used to fit the model. If None, the trainer provided in the
            constructor will be used.

        Raises
        ------
        RuntimeError
            If trainer is not provided either in the constructor or as an argument
            to the method.
        """
        self._check_if_model_exists()

        trainer = trainer if trainer is not None else self.trainer
        if trainer is None:
            raise RuntimeError(
                "No trainer provided. Trainer must be provided either "
                "in the constructor or as an argument to the partial_fit method."
            )

        trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs
        trainer_args = self.default_trainer_args.copy()
        trainer_args.update(trainer_kwargs)

        trainer.train(
            self.model,
            dataset,
            self.feature_transformer,
            self.label_transform_fn,
            **trainer_args,
        )

    def predict(self, dataset: Dataset, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Computes the prediction of the model for every example in the provided
        dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to compute predictions for.

        batch_size : int
            If None, predictions for the whole dataset will be done in a single batch.
            Else, predictions will be calculated in batches of batch_size size.
            This argument is useful in case the whole dataset can't be processed in a
            single batch.

        kwargs
            Keyword arguments passed to the model's `predict` method

        Returns
        -------
        ndarray
            Tensor containing predictions for examples in the passed Dataset.
        """
        # TODO: new method of providing examples must be defined.
        # examples is taken in dataset form as proof-of-concept.
        self._check_if_model_exists()

        y = []
        prediction_key = AbstractSupervisedModel.PREDICTION_KEY

        if batch_size is None:
            x_batch_tensor = self.feature_transformer.transform(dataset.batch())
            batch_prediction = self.model.predict(x_batch_tensor, **kwargs)
            prediction_tensor = batch_prediction[prediction_key]
            return prediction_tensor
        else:
            prediction_iterator = Iterator(batch_size=batch_size, shuffle=False)

            for batch in prediction_iterator(dataset):
                x_batch_tensor = self.feature_transformer.transform(batch)
                batch_prediction = self.model.predict(x_batch_tensor, **kwargs)
                prediction_tensor = batch_prediction[prediction_key]
                y.append(prediction_tensor)

            return np.concatenate(y)

    def _check_if_model_exists(self):
        if self.model is None:
            raise RuntimeError(
                "Model instance not available. Please provide "
                "a model instance in the constructor or call `fit` "
                "before calling `partial_fit.`"
            )

    def __repr__(self):
        attrs = {"model": self.model, "trainer": self.trainer}
        return repr_type_and_attrs(self, attrs)
