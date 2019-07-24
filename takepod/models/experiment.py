import numpy as np

from typing import Tuple, Callable, NamedTuple, Dict

from takepod.storage import Iterator, SingleBatchIterator, Dataset
from takepod.models import AbstractSupervisedModel
from takepod.models.simple_trainers import AbstractTrainer


class Experiment:

    def __init__(self,
                 model: AbstractSupervisedModel,
                 trainer: AbstractTrainer,
                 batch_to_tensor:
                 Callable[[NamedTuple, NamedTuple], Tuple[np.ndarray, np.ndarray]],
                 training_iterator_callable: Callable[[Dataset], Iterator],
                 prediction_iterator_callable: Callable[[Dataset], Iterator] = None
                 ):
        self.model = model
        self.trainer = trainer
        self.training_iterator_callable = training_iterator_callable
        self.batch_to_tensor = batch_to_tensor

        if prediction_iterator_callable is None:
            def default_prediction_iterator_callable(dataset):
                return SingleBatchIterator(dataset)

            self.prediction_iterator_callable = default_prediction_iterator_callable

        else:
            self.prediction_iterator_callable = prediction_iterator_callable

    def fit(self,
            dataset: Dataset,
            model_kwargs: dict = {},
            trainer_kwargs: dict = {}
            ):
        self.model.reset(**model_kwargs)
        train_iterator = self.training_iterator_callable(dataset)
        self.trainer.train(self.model, train_iterator, self.batch_to_tensor, **trainer_kwargs)

    def predict(self,
                dataset: Dataset,
                **kwargs
                ) -> Dict:
        # TODO: new method of providing examples must be defined.
        # examples is taken in dataset form as proof-of-concept.

        x = []

        for x_batch, y_batch in self.prediction_iterator_callable(dataset):
            x_batch_tensor, _ = self.batch_to_tensor(x_batch, y_batch)
            x.extend(x_batch_tensor)

        x_tensor = np.vstack(x)

        return self.model.predict(x_tensor, **kwargs)
