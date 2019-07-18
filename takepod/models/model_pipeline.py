from takepod.storage import Iterator, SingleBatchIterator, Dataset
from takepod.models import AbstractSupervisedModel
from takepod.models.simple_trainers import AbstractTrainer


class ModelPipeline:

    def __init__(self,
                 model: AbstractSupervisedModel,
                 trainer: AbstractTrainer,
                 iterator: Iterator,
                 batch_to_tensor: callable
                 ):
        self.model = model
        self.trainer = trainer
        self.iterator = iterator
        self.batch_to_tensor = batch_to_tensor

    def fit(self,
            dataset: Dataset,
            model_kwargs: dict = {},
            trainer_kwargs: dict = {}
            ):
        self.model.reset(**model_kwargs)
        self.iterator.set_dataset(dataset)
        self.trainer.train(self.model, self.iterator, self.batch_to_tensor, **trainer_kwargs)

    def predict(self,
                dataset: Dataset,
                **kwargs
                ):
        # TODO: new method of providing examples must be defined.
        # examples is taken in dataset form as proof-of-concept.

        x_batch, y_batch = next(SingleBatchIterator(dataset).__iter__())
        x_batch, y_batch = self.batch_to_tensor(x_batch, y_batch)
        return self.model.predict(x_batch, **kwargs)