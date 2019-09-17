from takepod.datasets import Dataset, SingleBatchIterator
from takepod.models import AbstractSupervisedModel


class Pipeline:

    def __init__(self,
                 fields,
                 create_example_fn,
                 feature_transform_fn,
                 model,
                 predict_kwargs
                 ):
        self.model = model
        self.fields = fields
        self.create_example_fn = create_example_fn
        self.feature_transform_fn = feature_transform_fn
        self.predict_kwargs = predict_kwargs

    def predict(self, example):
        processed_example = self.create_example_fn(example)
        ds = Dataset([processed_example], self.fields)

        x_batch, _ = next(SingleBatchIterator(ds).__iter__())
        x = self.feature_transform_fn(x_batch)
        prediction_dict = self.model.predict(x, **self.predict_kwargs)
        return prediction_dict[AbstractSupervisedModel.PREDICTION_KEY]
