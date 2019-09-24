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
                 model,
                 predict_kwargs=None,
                 trainer=None,
                 trainer_kwargs=None,
                 trainer_iterator=None,
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

        self.model = model
        self.fields = fields
        self.example_format = example_format
        self.feature_transformer = feature_transformer
        self.predict_kwargs = predict_kwargs
        self.example_factory = ExampleFactory(fields)

        self.trainer = trainer
        self.trainer_kwargs = trainer_kwargs
        self.trainer_iterator = trainer_iterator
        self.model_args = model_args
        self.label_transform_fn = label_transform_fn

    def predict(self, raw_example):
        processed_example = self.example_factory.from_format(raw_example,
                                                             self.example_format)
        ds = Dataset([processed_example], self.fields)

        x_batch, _ = next(SingleBatchIterator(ds).__iter__())
        x = self.feature_transformer.transform(x_batch)
        prediction_dict = self.model.predict(x, **self.predict_kwargs)
        return prediction_dict[AbstractSupervisedModel.PREDICTION_KEY]

    def fit(self,
            dataset: Dataset,
            trainer=None,
            trainer_iterator=None,
            trainer_kwargs=None,
            model_kwargs=None,
            reset_model=True):
        trainer = trainer if trainer is not None else self.trainer
        if trainer is None:
            errmsg = "No trainer provided. Trainer must be provided either in the " \
                     "constructor or as an argument."
            _LOGGER.error(errmsg)
            raise RuntimeError(errmsg)

        trainer_kwargs = trainer_kwargs if trainer_kwargs is not None \
            else self.trainer_kwargs
        if trainer_kwargs is None:
            errmsg = "No trainer_kwargs provided. Trainer arguments must be provided " \
                     "either in the constructor or as an argument. If no arguments are " \
                     "necessary, please pass an empty dict."
            _LOGGER.error(errmsg)
            raise RuntimeError(errmsg)

        model_kwargs = model_kwargs if model_kwargs is not None \
            else self.model_kwargs
        if model_kwargs is None:
            errmsg = "No model_kwargs provided. Model arguments must be provided " \
                     "either in the constructor or as an argument. If no arguments are " \
                     "necessary, please pass an empty dict."
            _LOGGER.error(errmsg)
            raise RuntimeError(errmsg)

        trainer_iterator = trainer_iterator if trainer_iterator is not None \
            else self.trainer_iterator
        if trainer_iterator is None:
            errmsg = "No trainer_iterator provided. Trainer_iterator must be provided " \
                     "either in the constructor or as an argument. If no arguments are " \
                     "necessary, please pass an empty dict."
            _LOGGER.error(errmsg)
            raise RuntimeError(errmsg)

        experiment = Experiment(self.model,
                                trainer,
                                lambda ds: trainer_iterator.set_dataset(ds),
                                feature_transformer=self.feature_transformer,
                                label_transform_fun=self.label_transform_fn)
        if reset_model:
            experiment.fit(dataset,
                           model_kwargs,
                           trainer_kwargs)
        else:
            experiment.partial_fit(dataset,
                                   trainer_kwargs)

        self.model = experiment.model
        self.
