"""Module contains the converter class for processing the HuggingFace Datasets."""
import logging

from podium.datasets import Dataset
from podium.storage import ExampleFactory, Field, LabelField, Vocab
from podium.util import log_and_raise_error

_LOGGER = logging.getLogger(__name__)

try:
    import datasets
except ImportError as e:
    _LOGGER.error('Problem occured while trying to import datasets. '
                  'If the library is not installed visit '
                  'https://huggingface.co/docs/datasets/ for more details.')
    raise e


def _identity(x):
    return x


class _FeatureConverter:
    """Class for converting features of the HuggingFace Dataset to the podium.storage.Fields.

    Notes
    -----
    This class should not be used directly. Instead, use
    the `convert_features_to_fields` function for conversion.
    """

    @staticmethod
    def convert(name, feature):
        """Convert a feature to the podium.storage.Field.

        Parameters:
        name : str
            Name of the column corresponding to the feature.
        feature : datasets.features.FeatureType
            Column feature.

        Returns
        -------
        podium.storage.Field
            podium.storage.Field after the conversion.

        Raises
        ------
        TypeError
            If conversion of the given feature type is not supported.
        """
        if isinstance(feature, datasets.ClassLabel):
            field = LabelField(name=name,
                               numericalizer=_identity,
                               allow_missing_data=False)
            return field

        elif isinstance(feature, datasets.Value):
            dtype = feature.dtype

            if dtype in {'bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
                         'int32', 'int64', 'float16', 'float32', 'float64'}:
                kwargs = {'tokenizer': None,
                          'keep_raw': False,
                          'numericalizer': _identity}

            elif dtype in {'string', 'utf8'}:
                kwargs = {'numericalizer': Vocab()}

            else:
                # some dtypes are not processed and stored as-is
                # for the full list see:
                # https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions
                kwargs = {'tokenizer': None,
                          'keep_raw': False,
                          'numericalizer': None}

        elif isinstance(feature,
                        (dict, list, datasets.Sequence,
                         datasets.Translation,
                         datasets.TranslationVariableLanguages)):
            kwargs = {'tokenizer': None,
                      'keep_raw': False,
                      'numericalizer': None}

        else:
            error_msg = f'Conversion for feature type {type(feature).__name__} ' \
                        'is not supported'
            log_and_raise_error(TypeError, _LOGGER, error_msg)

        # allow missing data for all fields except
        # the ones corresponding to the datasets.ClassLabel
        kwargs.update({'allow_missing_data': True})
        field = Field(name=name, **kwargs)

        return field


def convert_features_to_fields(features):
    """Convert a dictionary that maps column names of the HuggingFace Dataset
    to the features into a dictionary that maps column names to the podium.storage.Fields.

    Parameters
    ----------
    features : dict(str, datasets.features.FeatureType)
        Dictionary that maps a column name to the feature.

    Returns
    -------
    dict(str, podium.storage.Field)
        Dictionary that maps a column name to the podium.storage.Field.
    """
    return {
        name: _FeatureConverter.convert(name, feature)
        for name, feature in features.items()
    }


class HuggingFaceDatasetConverter:
    """Class for converting rows from the HuggingFace Datasets to podium.storage.Example.
    """

    def __init__(self, dataset, fields=None):
        """HuggingFaceDatasetConverter constructor.

        Parameters
        ----------
        dataset : datasets.Dataset
            HuggingFace Dataset.

        fields : dict(str, podium.storage.Field)
            Dictionary that maps a column name of the dataset to the podium.storage.Field.
            If passed None the default feature conversion rules will be used
            to build a dictonary from the dataset features.

        Raises
        ------
        TypeError
            If dataset is not an instance of datasets.Dataset.
        """
        if not isinstance(dataset, datasets.Dataset):
            error_msg = 'Incorrect dataset type. Expected datasets.Dataset, ' \
                        f'but got {type(dataset).__name__}'
            log_and_raise_error(TypeError, _LOGGER, error_msg)

        self.dataset = dataset
        self.fields = fields or convert_features_to_fields(dataset.features)

    def __iter__(self):
        """Iterate through the dataset and convert the examples."""
        example_factory = ExampleFactory(self.fields)

        for raw_example in self.dataset:
            yield example_factory.from_dict(raw_example)

    def as_dataset(self):
        """Convert the original HuggingFace dataset to a podium.storage.Dataset.

        Returns
        -------
        podium.storage.Dataset
            podium.storage.Dataset instance.
        """
        return Dataset(list(self), self.fields)
