from enum import Enum, auto
import logging

from takepod.storage import ExampleFactory, ExampleFormat
from takepod.datasets import Dataset, SingleBatchIterator
from takepod.models import AbstractSupervisedModel, FeatureTransformer

_LOGGER = logging.getLogger(__name__)


class Pipeline:

    def __init__(self,
                 fields,
                 example_format: ExampleFormat,
                 feature_transformer: FeatureTransformer,
                 model,
                 predict_kwargs
                 ):
        if example_format in (ExampleFormat.LIST, ExampleFormat.CSV, ExampleFormat.NLTK):
            if not isinstance(fields, (list, tuple)):
                error_msg = "If example format is LIST, CSV or NLTK, `fields`" \
                            "must be either a list or tuple. " \
                            "Type of `fields`: {}".format(type(fields))
                _LOGGER.error(error_msg)
                raise TypeError(error_msg)
        elif not isinstance(fields, dict):
            error_msg = "If example format is DICT, XML or JSON, `fields`" \
                        "must be a dict. " \
                        "Type of `fields`: {}".format(type(fields))
            _LOGGER.error(error_msg)
            raise TypeError(error_msg)

        self.model = model
        self.fields = fields
        self.example_format = example_format
        self.feature_transformer = feature_transformer
        self.predict_kwargs = predict_kwargs
        self.example_factory = ExampleFactory(fields)

    def predict(self, raw_example):
        processed_example = self.example_factory.from_format(raw_example,
                                                             self.example_format)
        ds = Dataset([processed_example], self.fields)

        x_batch, _ = next(SingleBatchIterator(ds).__iter__())
        x = self.feature_transformer.transform(x_batch)
        prediction_dict = self.model.predict(x, **self.predict_kwargs)
        return prediction_dict[AbstractSupervisedModel.PREDICTION_KEY]
