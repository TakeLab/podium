from takepod.storage import SingleBatchIterator


class ModelPipeline:

    def __init__(self, model, trainer, iterator, batch_to_tensor):
        self.model = model
        self.trainer = trainer
        self.iterator = iterator
        self.batch_to_tensor = batch_to_tensor

    def fit(self, dataset, model_args, trainer_args):
        self.model.reset(**model_args)
        self.iterator.set_dataset(dataset)
        self.trainer.train(self.model, self.iterator, self.batch_to_tensor, **trainer_args)

    def predict(self, dataset, **kwargs):
        # TODO: examples is taken in dataset form as proof-of-concept.
        # new method of providing examples must be defined.

        x_batch, y_batch = next(SingleBatchIterator(dataset).__iter__())
        x_batch, y_batch = self.batch_to_tensor(x_batch, y_batch)
        return self.model.predict(x_batch, **kwargs)