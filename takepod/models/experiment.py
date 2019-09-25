"""Modules defines an experiment - class used to combine iteration over data,
model training and prediction."""
from typing import Callable, NamedTuple, Dict, Type, Union
from inspect import isclass
import logging

import numpy as np

from takepod.datasets.dataset import Dataset
from takepod.datasets.iterator import Iterator, SingleBatchIterator
from takepod.models import AbstractSupervisedModel, \
    default_feature_transform, default_label_transform, FeatureTransformer
from takepod.models.trainer import AbstractTrainer

_LOGGER = logging.getLogger(__name__)


class Experiment:
    """Class used to streamline model fitting and prediction."""

    def __init__(self,
                 model: Union[Type[AbstractSupervisedModel], AbstractSupervisedModel],
                 feature_transformer: FeatureTransformer = None,
                 trainer: AbstractTrainer = None,
                 training_iterator_callable: Callable[[Dataset], Iterator] = None,
                 prediction_iterator_callable: Callable[[Dataset], Iterator] = None,
                 label_transform_fun:
                 Callable[[NamedTuple], np.ndarray] = None
                 ):
        """Creates a new Experiment. The Experiment class is used to simplify model
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

        training_iterator_callable : Callable[[Dataset], Iterator]
            Callable used to instantiate new instances of the Iterator used in fitting the
            model.

        prediction_iterator_callable : Callable[[Dataset], Iterator]
            Callable used to instantiate new instances of the Iterator used in prediction.
            Tensors which are prediction results for seperate batches will be stacked into
            a single tensor before being returned. If passed None, a SingleBatchIterator
            will be used as a default.

        feature_transformer : FeatureTransformer
            FeatureTransformer that transforms the input part of the batch returned by the
            iterator into features that can be fed into the model. Will also be fitted
            during Experiment fitting.

        label_transform_fun : Callable[[NamedTuple], np.ndarray]
            Callable that transforms the target part of the batch returned by the iterator
            into the same format the model prediction is. For a hypothetical perfect model
            the prediction result of the model for some examples must be identical to the
            result of this callable for those same examples.
        """
        if isclass(model):
            self.model_class = model
            self.model = None
        else:
            self.model_class = model.__class__
            self.model = model

        self.trainer = trainer
        self.training_iterator_callable = training_iterator_callable

        self.set_default_model_args()
        self.set_default_trainer_args()

        if prediction_iterator_callable is None:
            def default_prediction_iterator_callable(dataset):
                return SingleBatchIterator(dataset)

            self.prediction_iterator_callable = default_prediction_iterator_callable
        else:
            self.prediction_iterator_callable = prediction_iterator_callable

        self.feature_transformer = feature_transformer \
            if feature_transformer is not None \
            else FeatureTransformer(default_feature_transform)

        self.label_transform_fun = label_transform_fun \
            if label_transform_fun is not None \
            else default_label_transform

    def set_default_model_args(self, **kwargs):
        """Sets the default model arguments. Model arguments are keyword arguments passed
        to the model constructor. Default arguments can be updated/overridden by arguments
        in the `model_kwargs` dict in the fit method.

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

    def fit(self,
            dataset: Dataset,
            model_kwargs: Dict = None,
            trainer_kwargs: Dict = None,
            feature_transformer: FeatureTransformer = None,
            trainer: AbstractTrainer = None,
            training_iterator_callable: Callable[[Dataset], Iterator] = None,
            ):
        """Fits the model to the provided Dataset. During fitting, the provided Iterator
        and Trainer are used.

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
        """

        model_kwargs = {} if model_kwargs is None else model_kwargs
        trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs

        model_args = self.default_model_args.copy()
        model_args.update(model_kwargs)

        trainer_args = self.default_trainer_args.copy()
        trainer_args.update(trainer_kwargs)

        trainer = trainer if trainer is not None else self.trainer
        if trainer is None:
            errmsg = "No trainer provided. Trainer must be provided either in the " \
                     "constructor or as an argument to the fit method."
            _LOGGER.error(errmsg)
            raise RuntimeError(errmsg)

        if feature_transformer is not None:
            self.feature_transformer = feature_transformer

        # Fit the feature transformer if it needs fitting
        if self.feature_transformer.requires_fitting():
            x_batch, y_batch = next(SingleBatchIterator(dataset).__iter__())
            y = self.label_transform_fun(y_batch)
            self.feature_transformer.fit(x_batch, y)

        # Create new model instance
        self.model = self.model_class(**model_args)

        training_iterator_callable = training_iterator_callable \
            if training_iterator_callable is not None \
            else self.training_iterator_callable

        train_iterator = training_iterator_callable(dataset)
        # Train the model
        trainer.train(self.model,
                      train_iterator,
                      self.feature_transformer,
                      self.label_transform_fun,
                      **trainer_args)

    def partial_fit(self,
                    dataset: Dataset,
                    trainer_kwargs: Dict = None,
                    trainer: AbstractTrainer = None,
                    training_iterator_callable: Callable[[Dataset], Iterator] = None):
        self._check_if_model_exists()

        trainer = trainer if trainer is not None else self.trainer
        if trainer is None:
            errmsg = "No trainer provided. Trainer must be provided either in the " \
                     "constructor or as an argument to the partial_fit method."
            _LOGGER.error(errmsg)
            raise RuntimeError(errmsg)

        trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs
        trainer_args = self.default_trainer_args.copy()
        trainer_args.update(trainer_kwargs)

        training_iterator_callable = training_iterator_callable \
            if training_iterator_callable is not None \
            else self.training_iterator_callable

        train_iterator = training_iterator_callable(dataset)

        trainer.train(self.model,
                      train_iterator,
                      self.feature_transformer,
                      self.label_transform_fun,
                      **trainer_args)

    def predict(self,
                dataset: Dataset,
                **kwargs
                ) -> np.ndarray:
        """Computes the prediction of the model for every example in the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to compute predictions for.

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

        for x_batch, _ in self.prediction_iterator_callable(dataset):
            x_batch_tensor = self.feature_transformer.transform(x_batch)
            batch_prediction = self.model.predict(x_batch_tensor, **kwargs)
            prediction_tensor = batch_prediction[AbstractSupervisedModel.PREDICTION_KEY]
            y.append(prediction_tensor)

        return np.concatenate(y)

    def _check_if_model_exists(self):
        if self.model is None:
            errmsg = "Model instance not available. Please provide a model instance in " \
                     "the constructor or call `fit` before calling `partial_fit.`"
            _LOGGER.error(errmsg)
            raise RuntimeError(errmsg)
