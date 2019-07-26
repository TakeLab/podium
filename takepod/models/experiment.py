import numpy as np

from typing import Tuple, Callable, NamedTuple, Dict, Type

from takepod.storage import Iterator, SingleBatchIterator, Dataset
from takepod.models import AbstractSupervisedModel
from takepod.models.simple_trainers import AbstractTrainer


class Experiment:
    """Class used to streamline model fitting and prediction."""

    def __init__(self,
                 model_class: Type[AbstractSupervisedModel],
                 trainer: AbstractTrainer,
                 batch_to_tensor:
                 Callable[[NamedTuple, NamedTuple], Tuple[np.ndarray, np.ndarray]],
                 training_iterator_callable: Callable[[Dataset], Iterator],
                 prediction_iterator_callable: Callable[[Dataset], Iterator] = None,
                 ):
        """Creates a new Experiment. The Experiment class is used to simplify model 
        fitting and prediction using Podium components.
        
        Parameters
        ----------
        model_class : class
            Class of the Model to be fitted.
            Must be a subclass of Podium's `AbstractSupervisedModel`
            
        trainer : AbstractTrainer
            Trainer used to fit the model.
        
        batch_to_tensor : callable(x_batch, y_batch) -> x_tensor, y_tensor
            Callable used to transform Podium batches into a form the model uses.
            
        training_iterator_callable : callable(Dataset) -> Iterator
            Callable used to instantiate new instances of the Iterator used in fitting the
            model.
             
        prediction_iterator_callable : callable(Dataset) -> Iterator
            Callable used to instantiate new instances of the Iterator used in prediction.
            Tensors which are prediction results for seperate batches will be stacked into
            a single tensor before being returned. If passed None, a SingleBatchIterator
            will be used as a default.
        """
        self.model_class = model_class
        self.model = None
        self.trainer = trainer
        self.training_iterator_callable = training_iterator_callable
        self.batch_to_tensor = batch_to_tensor

        self.set_default_model_args()
        self.set_default_trainer_args()

        if prediction_iterator_callable is None:
            def default_prediction_iterator_callable(dataset):
                return SingleBatchIterator(dataset)

            self.prediction_iterator_callable = default_prediction_iterator_callable

        else:
            self.prediction_iterator_callable = prediction_iterator_callable

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
            trainer_kwargs: Dict = None
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
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs

        model_args = self.default_model_args.copy()
        model_args.update(model_kwargs)

        trainer_args = self.default_trainer_args.copy()
        trainer_args.update(trainer_kwargs)

        # Create new model instance
        self.model = self.model_class(**model_args)
        train_iterator = self.training_iterator_callable(dataset)

        self.trainer.train(self.model,
                           train_iterator,
                           self.batch_to_tensor,
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

        y = []

        for x_batch, y_batch in self.prediction_iterator_callable(dataset):
            x_batch_tensor, _ = self.batch_to_tensor(x_batch, y_batch)
            batch_prediction = self.model.predict(x_batch_tensor, **kwargs)
            prediction_tensor = batch_prediction[AbstractSupervisedModel.PREDICTION_KEY]
            y.extend(prediction_tensor)

        # TODO: always returns at least 2-D tensors example X labels
        # if lables are just one number (simple classification) maybe make it return a
        # 1D array? Will have to discuss a framework-wide convention.
        return np.vstack(y)
