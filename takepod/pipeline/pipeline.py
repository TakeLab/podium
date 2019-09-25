from typing import Union, Dict, List
import logging

from takepod.storage import ExampleFactory, ExampleFormat
from takepod.datasets import Dataset, SingleBatchIterator
from takepod.models import AbstractSupervisedModel, FeatureTransformer, Experiment

_LOGGER = logging.getLogger(__name__)


class Pipeline:

    def __init__(self,
                 fields: Union[Dict, List],
                 example_format: ExampleFormat,
                 feature_transformer: FeatureTransformer,
                 model: AbstractSupervisedModel,
                 trainer=None,
                 trainer_args=None,
                 trainer_iterator_callable=None,
                 model_args=None,
                 label_transform_fn=None
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

        self.fields = fields
        self.example_format = example_format
        self.example_factory = ExampleFactory(fields)

        self.experiment = Experiment(model,
                                     feature_transformer=feature_transformer,
                                     trainer=trainer,
                                     training_iterator_callable=trainer_iterator_callable,
                                     label_transform_fun=label_transform_fn)
        self.experiment.set_default_model_args(**model_args)
        self.experiment.set_default_trainer_args(**trainer_args)

    def predict_raw(self, raw_example):
        processed_example = self.example_factory.from_format(raw_example,
                                                             self.example_format)
        ds = Dataset([processed_example], self.fields)

        return self.experiment.predict(ds)

    def predict(self, dataset):
        self.experiment.predict(dataset)

    def fit(self,
            dataset: Dataset,
            feature_transformer=None,
            trainer=None,
            trainer_iterator_callable=None,
            trainer_kwargs=None,
            model_kwargs=None):
        self.experiment.fit(dataset,
                            model_kwargs=model_kwargs,
                            trainer_kwargs=trainer_kwargs,
                            feature_transformer=feature_transformer,
                            trainer=trainer,
                            training_iterator_callable=trainer_iterator_callable
                            )

    def partial_fit(self,
                    dataset: Dataset,
                    trainer=None,
                    trainer_iterator_callable=None,
                    trainer_kwargs=None):
        self.experiment.partial_fit(dataset,
                                    trainer_kwargs=trainer_kwargs,
                                    trainer=trainer,
                                    trainer_iterator_callable=trainer_iterator_callable)
