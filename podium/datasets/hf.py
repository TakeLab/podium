"""
Module contains the converter class for processing the HuggingFace Datasets.
"""
import textwrap
from typing import Dict, Iterator, Optional, Union

from podium.datasets import Dataset, DatasetBase
from podium.field import Field, LabelField
from podium.utils.general_utils import repr_type_and_attrs
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


def _convert_feature(name: str, feature: datasets.features.FeatureType) -> Field:
    """
    Function for converting features of the HuggingFace Dataset to the
    podium.Fields.

    Parameters
    ----------
    name : str
        Name of the column corresponding to the feature.
    feature : datasets.features.FeatureType
        Column feature.

    Returns
    -------
    podium.Field
        podium.Field after the conversion.

    Raises
    ------
    TypeError
        If conversion of the given feature type is not supported.
    """
    if isinstance(feature, datasets.ClassLabel):
        field = LabelField(name=name, numericalizer=_identity, allow_missing_data=False)
        return field

    elif isinstance(feature, datasets.Value):
        dtype = feature.dtype

        if dtype in {
            "bool",
            "binary",
            "large_binary",
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
            "float",  # alias for float32
            "double",  # alias for float64
        }:
            kwargs = {
                "tokenizer": None,
                "keep_raw": False,
                "numericalizer": _identity,
            }

        elif dtype in {"string", "large_string", "utf8"}:
            # Since the dataset won't actually be loaded, we have to
            # set eager to False here so .finalize_fields() triggers
            # the Vocab construction later.
            kwargs = {"numericalizer": Vocab(eager=False)}

        else:
            # the other dtypes are not processed and are stored as-is,
            # for the full list see: https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions
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
            f"Conversion for feature type {type(feature).__name__} " "is not supported"
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
    the features into a dictionary that maps column names to podium.Fields.

    Parameters
    ----------
    features : dict(str, datasets.features.FeatureType)
        Dictionary that maps a column name to the feature.

    Returns
    -------
    dict(str, podium.Field)
        Dictionary that maps a column name to a podium.Field.
    """
    return {name: _convert_feature(name, feature) for name, feature in features.items()}


class HFDatasetConverter(DatasetBase):
    """
    Class for converting rows from the HuggingFace Datasets to podium.Examples.
    """

    def __init__(
        self, hf_dataset: datasets.Dataset, fields: Optional[Dict[str, Field]] = None
    ) -> None:
        """
        HFDatasetConverter constructor.

        Parameters
        ----------
        hf_dataset : datasets.Dataset
            HuggingFace Dataset.

        fields : dict(str, podium.Field)
            Dictionary that maps a column name of the dataset to a podium.Field.
            If passed None the default feature conversion rules will be used
            to build a dictonary from the dataset features.

        Raises
        ------
        TypeError
            If dataset is not an instance of datasets.Dataset.
        """
        if not isinstance(hf_dataset, datasets.Dataset):
            raise TypeError(
                "Incorrect dataset type. Expected datasets.Dataset, "
                f"but got {type(hf_dataset).__name__}"
            )

        super().__init__(fields or convert_features_to_fields(hf_dataset.features))
        self._dataset = hf_dataset
        self._example_factory = ExampleFactory(self.field_dict)

    @property
    def dataset(self):
        return self._dataset

    def _get_examples(self):
        yield from self

    def __getitem__(self, i):
        raw_examples = self.dataset[i]

        # Index or slice
        if isinstance(i, int):
            return self._example_factory.from_dict(raw_examples)
        else:
            # Slice of hf.datasets.Dataset is a dictionary that maps
            # to a list of values. To map this to a list of our examples,
            # we map the single dictionary to a list of dictionaries and
            # then convert this to a list of podium Examples

            # Unpack the dict, creating a dict for each value tuple
            raw_examples = [
                {k: v for k, v in zip(raw_examples, values)}
                for values in zip(*raw_examples.values())
            ]

            # Map each raw example to a Podium example
            examples = [
                self._example_factory.from_dict(raw_example)
                for raw_example in raw_examples
            ]

            # Cast to a dataset
            return Dataset(examples, self.fields, sort_key=None)

    def __iter__(self) -> Iterator[Example]:
        """
        Iterate through the dataset and convert the examples.
        """
        for raw_example in self._dataset:
            yield self._example_factory.from_dict(raw_example)

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self):
        fields_str = ",\n".join(textwrap.indent(repr(f), " " * 8) for f in self.fields)
        fields_str = f"[\n{fields_str}\n    \n]"
        attrs = {
            "dataset_name": self._dataset.builder_name,
            "size": len(self),
            "fields": fields_str,
        }
        return repr_type_and_attrs(self, attrs, with_newlines=True, repr_values=False)

    def as_dataset(self) -> Dataset:
        """
        Convert the original HuggingFace dataset to a podium.Dataset.

        Returns
        -------
        podium.Dataset
            podium.Dataset instance.
        """
        return Dataset(list(self), self.fields)

    @staticmethod
    def from_dataset_dict(
        dataset_dict: Dict[str, datasets.Dataset],
        fields: Optional[Dict[str, Field]] = None,
        cast_to_podium: bool = False,
    ) -> Dict[str, Union["HFDatasetConverter", Dataset]]:
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

        def cast(dataset):
            dataset = HFDatasetConverter(dataset, fields)
            return dataset.as_dataset() if cast_to_podium else dataset

        return {name: cast(dataset) for name, dataset in dataset_dict.items()}
