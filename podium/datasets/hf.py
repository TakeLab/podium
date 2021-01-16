"""
Module contains the converter class for processing the HuggingFace Datasets.
"""
from typing import Dict, Iterator, Optional, Union

import podium
from podium.datasets import Dataset, DatasetBase
from podium.field import Field, LabelField
from podium.vocab import Vocab

from .example_factory import Example, ExampleFactory


try:
    import datasets
except ImportError:
    print(
        "Problem occured while trying to import datasets. "
        "If the library is not installed visit "
        "https://huggingface.co/docs/datasets/ for more details."
    )
    raise


def _identity(x):
    return x


class _FeatureConverter:
    """
    Class for converting features of the HuggingFace Dataset to the
    podium.storage.Fields.

    Notes
    -----
    This class should not be used directly. Instead, use
    the `convert_features_to_fields` function for conversion.
    """

    @staticmethod
    def convert(name: str, feature: datasets.features.FeatureType) -> Field:
        """
        Convert a feature to the podium.storage.Field.

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
            field = LabelField(
                name=name, numericalizer=_identity, allow_missing_data=False
            )
            return field

        elif isinstance(feature, datasets.Value):
            dtype = feature.dtype

            if dtype in {
                "bool",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "int8",
                "int16",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
            }:
                kwargs = {
                    "tokenizer": None,
                    "keep_raw": False,
                    "numericalizer": _identity,
                }

            elif dtype in {"string", "utf8"}:
                # Since the dataset won't _actually_ be loaded, we have to
                # set eager to False here so .finalize_fields() triggers
                # the Vocab construction later.
                kwargs = {"numericalizer": Vocab(eager=False)}

            else:
                # some dtypes are not processed and stored as-is
                # for the full list see:
                # https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions
                kwargs = {"tokenizer": None, "keep_raw": False, "numericalizer": None}

        elif isinstance(
            feature,
            (
                dict,
                list,
                datasets.Sequence,
                datasets.Translation,
                datasets.TranslationVariableLanguages,
            ),
        ):
            kwargs = {"tokenizer": None, "keep_raw": False, "numericalizer": None}

        else:
            TypeError(
                f"Conversion for feature type {type(feature).__name__} "
                "is not supported"
            )

        # allow missing data for all fields except
        # the ones corresponding to the datasets.ClassLabel
        kwargs.update({"allow_missing_data": True})
        field = Field(name=name, **kwargs)

        return field


def convert_features_to_fields(
    features: Dict[str, datasets.features.FeatureType]
) -> Dict[str, Field]:
    """
    Convert a dictionary that maps column names of the HuggingFace Dataset to
    the features into a dictionary that maps column names to the
    podium.storage.Fields.

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


class HFDatasetConverter(DatasetBase):
    """
    Class for converting rows from the HuggingFace Datasets to
    podium.storage.Example.
    """

    def __init__(
        self, dataset: datasets.Dataset, fields: Optional[Dict[str, Field]] = None
    ) -> None:
        """
        HFDatasetConverter constructor.

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
            raise TypeError(
                "Incorrect dataset type. Expected datasets.Dataset, "
                f"but got {type(dataset).__name__}"
            )

        self._dataset = dataset
        self._fields = fields or convert_features_to_fields(dataset.features)
        self._example_factory = ExampleFactory(self._fields)

    @property
    def fields(self):
        return list(self._fields.values())

    @property
    def dataset(self):
        return self._dataset

    def _get_examples(self):
        yield from self

    def __getitem__(self, i):
        # Index or slice
        if isinstance(i, int):
            raw_example = self.dataset[i]
            print("Raw", type(raw_example), raw_example)
            return self._example_factory.from_dict(raw_example)
        else:
            # Slice of a datasets.Dataset is a dictionary that maps
            # to a list of values. To map this to a list of our examples,
            # we need to find out the length of the slice.
            raw_examples = self.dataset[i]

            # Determing the length of the subset
            size = len(next(iter(raw_examples.values())))

            examples = self._example_factory.from_dict_list(raw_examples, size=size)
            return Dataset(examples, self.fields, sort_key=None)

    def __iter__(self) -> Iterator[Example]:
        """
        Iterate through the dataset and convert the examples.
        """
        for raw_example in self._dataset:
            yield self._example_factory.from_dict(raw_example)

    def __len__(self) -> int:
        return len(self.dataset)

    def as_dataset(self) -> Dataset:
        """
        Convert the original HuggingFace dataset to a podium.storage.Dataset.

        Returns
        -------
        podium.storage.Dataset
            podium.storage.Dataset instance.
        """
        return Dataset(list(self), self.fields)

    @staticmethod
    def from_dataset_dict(
        dataset_dict: Dict[str, datasets.Dataset],
        cast_to_podium: bool = True,
    ) -> Dict[str, Union["HFDatasetConverter", podium.Dataset]]:
        """
        Copies the keys of given dictionary and converts the corresponding
        HuggingFace Datasets to the HFDatasetConverter instances.

        Parameters
        ----------
        dataset_dict: dict(str, datasets.Dataset)
            Dictionary that maps dataset names to HuggingFace Datasets.

        cast_to_podium: bool
            Determines whether to immediately convert the HuggingFace dataset
            to Podium dataset (if True), or shallowly wrap the HuggingFace dataset
            in the HFDatasetConverter class.
            The HFDatasetConverter class currently doesn't support full Podium
            functionality and will not work with other components in the library.

        Returns
        -------
        dict(str, Union[HFDatasetConverter, podium.Dataset])
            Dictionary that maps dataset names to HFDatasetConverter instances.

        Raises
        ------
        TypeError
            If the given argument is not a dictionary.
        """
        if not isinstance(dataset_dict, dict):
            raise TypeError(
                "Incorrect argument type. Expected dict, "
                f"but got {type(dataset_dict).__name__}"
            )

        dataset_dict = {
            dataset_name: HFDatasetConverter(dataset)
            for dataset_name, dataset in dataset_dict.items()
        }

        if cast_to_podium:
            dataset_dict = {
                name: dataset.as_dataset()
                for name, dataset in dataset_dict.items()
            }
            a_dataset = next(dataset_dict.values())
            a_dataset.finalize_fields()

        return dataset_dict
