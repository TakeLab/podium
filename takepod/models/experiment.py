import numpy as np

from typing import Tuple, Callable, NamedTuple, Dict, Type

from takepod.storage import Iterator, SingleBatchIterator, Dataset
from takepod.models import AbstractSupervisedModel
from takepod.models.simple_trainers import AbstractTrainer


class Experiment:

    def __init__(self,
                 model_class: Type[AbstractSupervisedModel],
                 trainer: AbstractTrainer,
                 batch_to_tensor:
                 Callable[[NamedTuple, NamedTuple], Tuple[np.ndarray, np.ndarray]],
                 training_iterator_callable: Callable[[Dataset], Iterator],
                 prediction_iterator_callable: Callable[[Dataset], Iterator] = None,
                 ):
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
        self.default_model_args = kwargs

    def set_default_trainer_args(self, **kwargs):
        self.default_trainer_args = kwargs

    def fit(self,
            dataset: Dataset,
            model_kwargs: dict = {},
            trainer_kwargs: dict = {}
            ):
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
        # TODO: new method of providing examples must be defined.
        # examples is taken in dataset form as proof-of-concept.

        y = []

        for x_batch, y_batch in self.prediction_iterator_callable(dataset):
            x_batch_tensor, _ = self.batch_to_tensor(x_batch, y_batch)
            batch_prediction = self.model.predict(x_batch_tensor, **kwargs)[AbstractSupervisedModel.PREDICTION_KEY]
            y.extend(batch_prediction)

        return np.vstack(y)
