from typing import Union, Dict, List, Callable, NamedTuple, Any, Type
import logging

from takepod.storage import ExampleFactory, ExampleFormat
from takepod.datasets import Dataset, Iterator
from takepod.models import AbstractSupervisedModel, FeatureTransformer, Experiment, \
    AbstractTrainer
import numpy as np

_LOGGER = logging.getLogger(__name__)


class Pipeline(Experiment):
    """Class used to streamline the use of Podium. It contains all components needed to
    train or fine-tune a pre-configured model and make predictions on new data."""

    def __init__(self,
                 fields: Union[Dict, List],
                 example_format: Union[ExampleFormat, str],
                 feature_transformer: FeatureTransformer,
                 model: Union[AbstractSupervisedModel, Type[AbstractSupervisedModel]],
                 trainer: AbstractTrainer = None,
                 trainer_iterator_callable: Callable[[Dataset], Iterator] = None,
                 label_transform_fn: Callable[[NamedTuple], np.ndarray] = None,
                 output_transform_fn: Callable[[np.ndarray], Any] = None
                 ):
        """Creates a new pipeline instance.

        Parameters
        ----------
        fields : dict or list of fields
            Fields used to process raw data.  Can be either a dict mapping column names
            to Fields (or tuples of Fields), or a list of Fields (or tuples of Fields).
            A Field value of None means the corresponding column will
            be ignored.

        example_format: ExampleFormat
            Format of expected raw examples.

        feature_transformer: FeatureTransformer
            FeatureTransformer used to transform data features from the podium "batch"
            format into numpy arrays. Will be fitted along with the model to the provided
            data.

        model : class or model instance
            Class of the Model to be fitted or a pre-trained model.
            If pre-trained model is passed and `fit` is called a new model instance will
            be created. For fine-tuning of the passed model instance call
            `partial_fit`.
            Must be a subclass of Podium's `AbstractSupervisedModel`

        trainer: AbstractTrainer, Optional
            Trainer used to fit the model. If provided, this trainer instance will be
            stored in the pipeline and used as the default trainer if no trainer is
            provided in the `fit` and `partial_fit` methods.

        trainer_iterator_callable: Callable[[Dataset], Iterator]
            Callable used to instantiate new instances of the Iterator used in fitting the
            model.

        label_transform_fn: Callable[[NamedTuple], np.ndarray]
            Callable that transforms the target part of the batch returned by the iterator
            into the same format the model prediction is. For a hypothetical perfect model
            the prediction result of the model for some examples must be identical to the
            result of this callable for those same examples.

        """
        if isinstance(example_format, ExampleFormat):
            example_format = example_format.value

        if example_format in (ExampleFormat.LIST.value, ExampleFormat.CSV.value,
                              ExampleFormat.NLTK.value):
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

        self.output_transform_fn = output_transform_fn

        super().__init__(model,
                         feature_transformer=feature_transformer,
                         trainer=trainer,
                         training_iterator_callable=trainer_iterator_callable,
                         label_transform_fun=label_transform_fn)

    def predict_raw(self,
                    raw_example: Any,
                    **kwargs) -> np.ndarray:
        """Computes the prediction of the model for the one example.
        The example must be of the format provided in the constructor as the
        `example_format` parameter.

        Parameters
        ----------
        raw_example: Any
            Example to compute the prediction for.

        kwargs
            Keyword arguments passed to the model's `predict` method

        Returns
        -------
        ndarray
            Tensor containing the prediction for the example."""
        processed_example = self.example_factory.from_format(raw_example,
                                                             self.example_format)
        ds = Dataset([processed_example], self.fields)
        prediction = self.predict(ds, **kwargs)
        prediction = prediction[0]

        if self.output_transform_fn is not None:
            return self.output_transform_fn(prediction)

        else:
            return prediction
