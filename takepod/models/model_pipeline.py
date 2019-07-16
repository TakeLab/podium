from abc import ABC

class ModelPipeline(ABC):

    def __init__(self, model, trainer, iterator, batch_to_tensor):
        self.model = model
        self.trainer = trainer
        self.iterator = iterator
        self.batch_to_tensor = batch_to_tensor

    def fit(self, dataset, **kwargs):
        self.model.reset()
        # TODO
