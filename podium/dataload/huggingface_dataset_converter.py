"""Module contains the converter class for processing the HuggingFace Datasets."""
import logging

from podium.datasets import Dataset
from podium.storage import ExampleFactory, Field, LabelField, Vocab

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
    """Class for converting features to the Podium fields.

    Notes
    -----
    This class should not be used directly. Instead, use
    the `convert_features_to_fields` function for conversion.
    """

    @staticmethod
    def convert(name, feature):
        """Convert a feature to the Podium field.

        Parameters:
        name : str
            Name of the column corresponding to the feature.
        feature: datasets.features.FeatureType
            Column feature.

        Returns
        -------
        Field
            Podium field after the conversion.

        Raises
        ------
        TypeError
            If conversion of the given feature type is not supported.
        """
        if isinstance(feature, datasets.ClassLabel):
            field = LabelField(name=name,
                               custom_numericalize=_identity)
            return field

        elif isinstance(feature, datasets.Value):
            dtype = feature.dtype

            if dtype in {'bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
                         'int32', 'int64', 'float16', 'float32', 'float64'}:
                kwargs = {'tokenize': False,
                          'store_as_raw': True,
                          'custom_numericalize': _identity}

            elif dtype in {'string', 'utf8'}:
                kwargs = {'vocab': Vocab()}

            else:
                kwargs = {'tokenize': False,
                          'store_as_raw': True,
                          'is_numericalizable': False}

        elif isinstance(feature,
                        (dict, list, datasets.Sequence,
                         datasets.Translation,
                         datasets.TranslationVariableLanguages)):
            kwargs = {'tokenize': False,
                      'store_as_raw': True,
                      'is_numericalizable': False}

        else:
            error_msg = 'Conversion for feature type {} is not supported' \
                        .format(type(feature).__name__)
            _LOGGER.error(error_msg)
            raise TypeError(error_msg)

        # allow missing data for all fields except
        # the ones corresponding to the datasets.ClassLabel
        kwargs.update({'allow_missing_data': True})
        field = Field(name=name, **kwargs)

        return field


def convert_features_to_fields(features):
    """Convert a dictionary that maps column names to the features
    into a dictionary that maps column names to the Podium fields.

    Parameters
    ----------
    features : dict(str, datasets.features.FeatureType)
        Dictioanry that maps a column name to the feature.

    Returns
    -------
    dict(str, Field)
        Dictionary that maps a column name to the Podium field.
    """
    return {
        name: _FeatureConverter.convert(name, feature)
        for name, feature in features.items()
    }


class HuggingFaceDatasetConverter:
    """Class for converting rows from the HuggingFace Datasets to Podium Examples."""

    def __init__(self, dataset, fields=None):
        """HuggingFaceDatasetConverter constructor.

        Parameters
        ----------
        dataset : datasets.Dataset
            HuggingFace Dataset.

        fields : dict(str, Field)
            Dictionary that maps a column name of the dataset to the field.
            If passed None the default feature conversion rules will be used
            to build a dictonary from the dataset features.

        Raises
        ------
            If dataset type is incorrect.
        """
        if not isinstance(dataset, datasets.Dataset):
            error_msg = 'Incorrect dataset type. Expected datasets.Dataset, but got {}' \
                        .format(type(dataset).__name__)
            _LOGGER.error(error_msg)
            raise TypeError(error_msg)

        self.dataset = dataset
        self.fields = fields or convert_features_to_fields(dataset.features)

    def __iter__(self):
        """Iterate through the dataset and convert the examples."""
        example_factory = ExampleFactory(self.fields)

        for raw_example in self.dataset:
            yield example_factory.from_dict(raw_example)

    def as_dataset(self):
        """Convert the original HuggingFace dataset to a Podium Dataset.

        Returns
        -------
        Dataset
            Podium Dataset.
        """
        return Dataset(list(self), self.fields)
