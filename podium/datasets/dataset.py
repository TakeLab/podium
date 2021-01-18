"""
Module contains base classes for datasets.
"""
import copy
import itertools
import random
from abc import ABC, abstractmethod
from bisect import bisect_right
from itertools import chain, islice
from math import ceil
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)


try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np

from podium.field import Field, unpack_fields

from .example_factory import Example


FieldType = Optional[Union[Field, List[Field]]]


class Comparable(Protocol):
    """
    Protocol for annotating comparable types.
    """

    @abstractmethod
    def __lt__(self: "CT", other: "CT") -> bool:
        pass


CT = TypeVar("CT", bound=Comparable)


def _get_permutation(size, seed, generator):
    if seed is not None and generator is not None:
        raise ValueError(
            "Both `seed` and `generator` were provided. Please specify just one of them."
        )

    if generator is None or isinstance(generator, np.random.Generator):
        raise ValueError(
            "The provided generator must be an instance of numpy.random.Generator"
        )

    if generator is None:
        if seed is None:
            seed = np.random.get_state()[1][0]
            np.random.random()
        generator = np.random.default_rng(seed)

    return generator.permutation(size)


class DatasetBase(ABC):
    """
    Abstract base class for all datasets in Podium.
    """

    def __init__(self, fields: Union[Dict[str, FieldType], List[FieldType]]):
        self._fields = tuple(unpack_fields(fields))

    @property
    def fields(self) -> Tuple[Field]:
        """
        Tuple containing all fields of this dataset.
        """
        return self._fields

    @property
    def field_dict(self) -> Dict[str, Field]:
        """
        Dictionary containing all field names mapping to their respective
        Fields.
        """
        return {f.name: f for f in self.fields}

    @property
    def examples(self) -> List[Example]:
        """
        List containing all Examples.
        """
        return self._get_examples()

    def __iter__(self) -> Iterator[Example]:
        """
        Iterates over all examples in the dataset in order.

        Yields
        ------
        Example
            Yields examples in the dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, field: Union[str, Field]) -> Iterator[Tuple[Any, Any]]:
        """
        Returns an Iterator iterating over values of the field with the given
        name for every example in the dataset.

        Parameters
        ----------
        field : str or podium.Field
            The name of the field whose values are to be returned.

        Returns
        ------
            an Iterator iterating over values of the field with the given name
            for every example in the dataset.

        Raises
        ------
        AttributeError
            If there is no Field with the given name.
        """
        field_name = field.name if isinstance(field, Field) else field

        if field_name in self.field_dict:

            def attr_generator(_dataset, _field_name):
                for x in _dataset:
                    yield x[field_name]

            return attr_generator(self, field_name)
        else:
            raise AttributeError(f"Dataset has no field {field_name}.")

    def finalize_fields(self, *datasets: "DatasetBase") -> None:
        """
        Builds vocabularies of all the non-eager fields in the dataset, from the
        Dataset objects given as \\*args and then finalizes all the fields.

        Parameters
        ----------
        \\*datasets
            A variable number of DatasetBase objects from which to build the
            vocabularies for non-eager fields. If none provided, the
            vocabularies are built from this Dataset (self).
        """

        # if there are non-eager fields, we need to build their vocabularies
        fields_to_build = [f for f in self.fields if not f.eager and f.use_vocab]
        if fields_to_build:
            # there can be multiple datasets we want to iterate over
            data_sources = [ds for ds in datasets if isinstance(ds, DatasetBase)]

            # use self as a data source if no other given
            if not data_sources:
                data_sources.append(self)

            # for each example in each dataset,
            # update all non-eager fields
            for dataset in data_sources:
                for example in dataset:
                    for field in fields_to_build:
                        _, tokenized = example[field.name]
                        field.update_vocab(tokenized)

        for field in self.fields:
            field.finalize()

    def batch(self) -> Tuple[NamedTuple, NamedTuple]:
        """
        Creates an input and target batch containing the whole dataset. The
        format of the batch is the same as the batches returned by the.

        Returns
        -------
        input_batch, target_batch
                Two objects containing the input and target batches over
                the whole dataset.
        """
        # Imported here because of circular import
        from podium.datasets import SingleBatchIterator

        return next(iter(SingleBatchIterator(self, shuffle=False)))

    def sort(self, key: Callable[[Example], CT], reverse=False) -> "DatasetBase":
        """
        Creates a new DatasetBase instance in which all Examples are sorted
        according to the value returned by `key`.

        Parameters
        ----------
        key: callable
            specifies a function of one argument that is used to extract a comparison key
            from each Example.

        reverse: bool
            If set to True, then the list elements are sorted as if each comparison were
            reversed.

        Returns
        -------
        DatasetBase
            A new DatasetBase instance with sorted Examples.
        """

        def index_key(i):
            return key(self[i])

        indices = list(range(len(self)))
        indices.sort(key=index_key, reverse=reverse)
        return self[indices]

    def filter(self, predicate: Callable[[Example], bool]) -> "DatasetBase":
        """
        Filters examples with given predicate and returns a new DatasetBase
        instance containing those examples.

        Parameters
        ----------
        predicate : callable
            predicate should be a callable that accepts example as input and returns
            true if the example shouldn't be filtered, otherwise returns false

        Returns
        -------
        DatasetBase
            A new DatasetBase instance containing only the Examples for which `predicate`
            returned True.
        """
        indices = [i for i, example in enumerate(self) if predicate(example)]
        return self[indices]

    def shuffle(
        self, seed: Optional[int] = None, generator: Optional[np.random.Generator] = None
    ) -> "DatasetBase":
        """
        Creates a new DatasetBase instance containing all Examples, but in
        shuffled order.

        Parameters
        ----------
        seed : int, optional
            A seed used to initialized the default NumPy random Generator.
            Default: None.
        generator: np.random.Generator, optional
            NumPy random Generator to use to compute the permutation.
            Default: None.

        Returns
        -------
        DatasetBase
            A new DatasetBase instance containing all Examples, but in shuffled
            order.
        """
        return self[_get_permutation(len(self), seed, generator)]

    def __repr__(self):
        # Distribute field prints across lines for readability
        fields_as_str = "\n   ".join([repr(f) for f in self.fields])

        if len(self.fields) > 1:
            # Prepend newline only in case there's multiple fields
            fields_as_str = f"\n  ({fields_as_str})"

        fields_as_str = f"Fields:{fields_as_str}\n"

        return f"{type(self).__name__}[Size: {len(self)}, {fields_as_str}]"

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of examples in the dataset.

        Returns
        -------
        int
            The number of examples in the dataset.
        """
        pass

    @overload
    def __getitem__(self, i: int) -> Example:
        ...

    @overload
    def __getitem__(self, i: Union[slice, Iterable[int]]) -> "DatasetBase":
        ...

    @abstractmethod
    def __getitem__(
        self, i: Union[int, slice, Iterable[int]]
    ) -> Union[Example, "DatasetBase"]:
        """
        Returns an example or a new dataset containing the indexed examples.

        If indexed with an int, only the example at that position will be returned.
        If Indexed with a slice or iterable, all examples indexed by the object
        will be collected and a new dataset containing only those examples will be
        returned. The new dataset will contain copies of the old dataset's fields and
        will be identical to the original dataset, with the exception of the example
        number and ordering. See wiki for detailed examples.

        Examples in the returned Dataset are the same ones present in the
        original dataset. If a complete deep-copy of the dataset, or its slice,
        is needed please refer to the `get` method.

        Usage example:

            example = dataset[1] # Indexing by single integer returns a single example

            new_dataset = dataset[1:10] # Multi-indexing returns a new dataset containing
                                        # the indexed examples.

        Parameters
        ----------
        i : int or slice or iterable of ints
            Index used to index examples.

        Returns
        -------
        single example or Dataset
            If i is an int, a single example will be returned.
            If i is a slice or iterable, a copy of this dataset containing
            only the indexed examples will be returned.
        """
        pass

    @abstractmethod
    def _get_examples(self) -> List[Example]:
        """
        Returns a list containing all examples of this dataset.
        """
        pass


class Dataset(DatasetBase):
    """
    A general purpose container for datasets. A dataset is a shallow wrapper for
    a list of `Example` classes which store the instance data as well as the
    corresponding `Field` classes, which process the columns of each example.

    Attributes
    ----------
    examples : list
        A list containing the instances of the dataset as Example classes.
    fields : list
        A list of Field objects defining preprocessing for data fields of
        the dataset.
    """

    def __init__(self, examples, fields, sort_key=None):
        """
        Creates a dataset with the given examples and their fields.

        Parameters
        ----------
        examples : list
            A list of examples.
        fields : list
            A list of fields that the examples have been created with.
        sort_key : callable
            A key to use for sorting dataset examples, used for batching
            together examples with similar lengths to minimize padding.
        """
        self._examples = examples
        self._sort_key = sort_key
        super().__init__(fields)

    def __getitem__(
        self, i: Union[int, Iterable[int], slice]
    ) -> Union["Dataset", Example]:
        if isinstance(i, int):
            return self._examples[i]

        examples = (
            self.examples[i]
            if isinstance(i, slice)
            else [self._examples[idx] for idx in i]
        )
        return Dataset(examples, self._fields)

    def __len__(self) -> int:
        return len(self._examples)

    def _get_examples(self) -> List[Example]:
        return self._examples

    def sort(
        self, key: Callable[[Example], CT], reverse=False, inplace: bool = False
    ) -> "Dataset":
        """
        Creates a new DatasetBase instance in which all Examples are sorted
        according to the value returned by `key`.

        Parameters
        ----------
        key: callable
            specifies a function of one argument that is used to extract a comparison key
            from each Example.
        reverse: bool
            If set to True, then the list elements are sorted as if each comparison were
            reversed.
        inplace : bool
            If True, the dataset is sorted in-place and returned.

        Returns
        -------
        Dataset
            A new Dataset instance with sorted Examples.
        """

        def index_key(i):
            return key(self[i])

        indices = list(range(len(self)))
        indices.sort(key=index_key, reverse=reverse)

        if inplace:
            self._examples = [self._examples[idx] for idx in indices]
            return self

        return super().sort(key, reverse)

    def filter(
        self, predicate: Callable[[Example], bool], inplace: bool = False
    ) -> "Dataset":
        """
        Filters examples with given predicate and returns a new Dataset
        instance containing those examples. If inplace is True, the dataset is
        modified in-place and returned.

        Parameters
        ----------
        predicate : callable
            Predicate should be a callable that accepts example as input and returns
            true if the example shouldn't be filtered, otherwise returns false
        inplace : bool
            If True, the dataset is filtered in-place and returned.

        Returns
        -------
        Dataset
            A new or the original Dataset instance
            containing only the Examples for which `predicate` returned True.
        """
        if inplace:
            self._examples = [example for example in self if predicate(example)]
            return self

        return super().filter(predicate)

    def shuffle(
        self,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        inplace: bool = False,
    ) -> "Dataset":
        """
        Creates a new Dataset instance containing all Examples, but in
        shuffled order. If inplace is True, the dataset is
        modified in-place and returned.

        Parameters
        ----------
        seed : int, optional
            A seed used to initialized the default NumPy random Generator.
            Default: None.
        generator: np.random.Generator, optional
            NumPy random Generator to use to compute the permutation.
            Default: None.
        inplace : bool
            If True, the dataset is shuffled in-place and returned.

        Returns
        -------
        Dataset
            A new or the original Dataset instance containing all Examples, but in shuffled
            order.
        """
        if inplace:
            self._examples = [
                self._examples[idx]
                for idx in _get_permutation(len(self), seed, generator)
            ]
            return self

        return super().shuffle(seed, generator)

    def copy(self, copy_fields: bool = False):
        """
        Returns a Dataset instance with the copied examples. If `copy_fields` is true,
        the dataset fields are copied as well.

        Parameters
        ----------
        copy_fields : bool
            If True, the dataset fields are copied as well.

        Returns
        -------
        Dataset
            A copied Dataset.
        """
        return Dataset(
            copy.deepcopy(self._examples),
            copy.deepcopy(self._fields) if copy_fields else self._fields,
            sort_key=self._sort_key,
        )

    def split(
        self,
        split_ratio=0.7,
        stratified=False,
        strata_field_name=None,
        random_state=None,
        shuffle=True,
    ):
        """
        Creates train-(validation)-test splits from this dataset.

        The splits are new Dataset objects, each containing a part of this
        one's examples.

        Parameters
        ----------
        split_ratio : (float | list[float] | tuple[float])
            If type is float, a number in the interval (0.0, 1.0) denoting
            the amount of data to be used for the train split (the rest
            is used for test).
            If type is list or tuple, it should be of length 2 (or 3) and
            the numbers should denote the relative sizes of train, (valid)
            and test splits respectively.
            If the relative size for valid is missing (length is 2), only
            the train-test split is returned (valid is taken to be 0.0).
            Also, the relative sizes don't have to sum up to 1.0 (they are
            normalized automatically).
            The ratio must not be so unbalanced that it would result in
            either of the splits being empty (having zero elements).
            Default is 0.7 (for the train set).
        stratified : bool
            Whether the split should be stratified. A stratified split
            means that for each concrete value of the strata field, the
            given train-val-test ratio is preserved. Usually used on
            fields representing labels / classes, so that every class is
            present in each of our splits with the same percentage as in
            the entire dataset.
            Default is False.
        strata_field_name : str
            Name of the field that is to be used to do the stratified
            split. Only relevant when 'stratified' is true.
            If the name of the strata field is not provided (the default
            behaviour), the stratified split will be done over the first
            field that is a target (its 'is_target' attribute is True).
            Note that the values of the strata field have to be hashable.
            Default is None.
        random_state : int
            The random seed used for shuffling.

        Returns
        -------
        tuple[Dataset]
            Datasets for train, (validation) and test splits in that order,
            depending on the split ratios that were provided.

        Raises
        ------
        ValueError
            If the given split ratio is not in one of the valid forms.
            If the given split ratio is in a valid form, but wrong in the
            sense that it would result with at least one empty split.
            If stratified is True and the field with the given
            strata_field_name doesn't exist.
        """
        train_ratio, val_ratio, test_ratio = check_split_ratio(split_ratio)

        # For the permutations
        random.seed(random_state)

        if not stratified:
            train_data, val_data, test_data = rationed_split(
                self.examples, train_ratio, val_ratio, test_ratio, shuffle
            )
        else:
            strata_field_name = self._get_strata_field_name(strata_field_name)

            if strata_field_name is None:
                raise ValueError(
                    "If strata_field_name is not provided, at least"
                    " one field has to have is_target equal to True."
                )

            if strata_field_name not in self.field_dict:
                raise ValueError(f"Invalid strata field name: {strata_field_name}")

            train_data, val_data, test_data = stratified_split(
                self.examples,
                train_ratio,
                val_ratio,
                test_ratio,
                strata_field_name,
                shuffle,
            )

        splits = tuple(
            Dataset(example_list, self.fields, sort_key=self._sort_key)
            for example_list in (train_data, val_data, test_data)
            if example_list
        )

        return splits

    def _get_strata_field_name(self, strata_field_name):
        if strata_field_name is not None:
            return strata_field_name

        for field in self.fields:
            if field.is_target:
                return field.name

        return None

    def numericalize_examples(self):
        """
        Generates and caches numericalized data for every example in the
        dataset.

        Call before using the dataset to avoid lazy numericalization during
        iteration.
        """
        for example in self.examples:
            for field in self.fields:
                # Generate and cache the numericalized data
                # the return value is ignored
                field.get_numericalization_for_example(example)

    def __getstate__(self):
        """
        Method obtains dataset state. It is used for pickling dataset data to
        file.

        Returns
        -------
        state : dict
            dataset state dictionary
        """
        return self.__dict__

    def __setstate__(self, state):
        """
        Method sets dataset state. It is used for unpickling dataset data from
        file.

        Parameters
        ----------
        state : dict
            dataset state dictionary
        """
        self.__dict__ = state

    @staticmethod
    def from_dataset(dataset: DatasetBase) -> "Dataset":
        """
        Creates an Dataset instance from a podium.datasets.DatasetBase instance.

        Parameters
        ----------
        dataset: DatasetBase
            DatasetBase instance to be used to create the Dataset.

        Returns
        -------
        Dataset
            Dataset instance created from the passed DatasetBase instance.
        """
        return Dataset(dataset.examples, dataset.fields)


class DatasetConcatView(DatasetBase):
    """
    View used for dataset concatenation.

    Two or more datasets can be concatenated. New fields can be provided as
    'field_overrides' that will be updated with all examples.
    """

    def __init__(
        self,
        datasets: List[DatasetBase],
        field_overrides: Optional[Union[Dict[str, Field], List[Field]]] = None,
    ):
        """
        View used for dataset concatenation.

        Two or more datasets can be concatenated. The concatenated dataset will
        contain only those fields which all datasets have in common, by name.

        New fields can be provided as 'field_overrides' that will be updated
        with all examples.

        Parameters
        ----------
        datasets: List[DatasetBase]
            A list datasets to be concatenated.
        field_overrides: Union[Dict[str, Field], List[Field]]
            A dict or list containing fields that will be used to override
            existing fields. Can be either a dict mapping old field names to new
            ones, or a list, in which case the field with the same name will be
            overridden. The overridden field will not be present in the
            concatenated view. The override field (if eager) will be updated
            with all examples from the concatenation.
        """
        if isinstance(datasets, DatasetBase):
            # Wrap single dataset in a list
            self._datasets = [datasets]
        elif isinstance(datasets, (list, tuple)):
            self._datasets = list(datasets)
        else:
            err_msg = (
                f"Invalid 'dataset' argument to {type(self).__name__}. "
                f"Supported values are lists or tuples of DatasetBase instances, "
                f"or a single DatasetBase instance. "
                f"Passed type: {type(datasets).__name__}"
            )
            raise TypeError(err_msg)

        if isinstance(field_overrides, list):
            field_overrides = {f.name: f for f in field_overrides}

        self._len = sum([len(ds) for ds in datasets])

        self._cumulative_lengths = [len(self._datasets[0])]
        for dataset in islice(datasets, 1, None):
            cumulative_len = self._cumulative_lengths[-1] + len(dataset)
            self._cumulative_lengths.append(cumulative_len)

        self._field_overrides = field_overrides or {}

        self._field_mapping = DatasetConcatView._create_field_mapping(
            self._datasets, self._field_overrides
        )
        self._reverse_field_name_mapping_dict = None

        fields = list(self._field_mapping.values())

        super().__init__(fields)
        self._update_override_fields()

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Example]:
        original_examples = chain(*self._datasets)
        mapped_examples = map(self._map_example, original_examples)
        yield from mapped_examples

    def __getattr__(self, field: Union[str, Field]) -> Iterator[Tuple[Any, Any]]:
        view_field_name = field if isinstance(field, str) else field.name
        if view_field_name not in self._reverse_field_name_mapping():
            # TODO better error message?
            err_msg = (
                f'Field "{view_field_name}" not present in this '
                f"{type(self).__name__}. "
                f"Fields: {list(f.name for f in self.fields)}"
            )
            raise ValueError(err_msg)

        original_field_name = self._reverse_field_name_mapping()[view_field_name]
        for ds in self._datasets:
            yield from getattr(ds, original_field_name)

    def __getitem__(self, item):

        if isinstance(item, int):
            dataset, index = self._translate_index(item)
            return self._map_example(dataset[index])

        else:
            return create_view(self, item)

    @staticmethod
    def _create_field_mapping(
        datasets: List[DatasetBase], field_overrides: Dict[str, Field]
    ) -> Dict[str, Field]:
        """
        Creates a dict mapping field names in the original datasets to fields
        they were overridden with: {"field_name" -> Field('override_field')}

        Parameters
        ----------
        datasets: List[DatasetBase]
            List of original datasets.

        field_overrides: Dict[str, Field]
            Dict mapping field names to override fields.

        Returns
        -------
        Dict[str, Field]
            A dict mapping all field names from the original datasets to Field
            instances. Respects field overrides.
        """
        intersection_field_names = DatasetConcatView._get_intersection_field_names(
            datasets
        )

        # Check for empty intersection
        if len(intersection_field_names) == 0:
            err_msg = (
                "Empty field name intersection. "
                "No field name is contained in all passed Datasets."
            )
            raise ValueError(err_msg)

        # Check for invalid overrides
        intersection_field_names_set = set(intersection_field_names)
        for fname in field_overrides:
            if fname not in intersection_field_names_set:
                err_msg = (
                    f'Override field name "{fname}" not contained in the '
                    f"intersection of passed datasets' fields: "
                    f"{intersection_field_names}"
                )
                raise ValueError(err_msg)

        # Check for vocab equality
        for fname in intersection_field_names:
            if fname in field_overrides:
                continue
            fields = [ds.field_dict[fname] for ds in datasets]
            vocabs = [f.vocab if f.use_vocab else None for f in fields]
            for i, v in enumerate(vocabs):
                if v != vocabs[0]:
                    raise ValueError(
                        f"Vocab inequality detected between datasets at positions 0 and "
                        f'{i} in field "{fname}". Vocab inequality can cause unexpected '
                        f"token indexing behavior during batching. Please ensure all "
                        f"datasets have equal Vocabs or provide override fields for "
                        f"mismatched Vocabs."
                    )

        field_mapping = {}
        default_field_dict = datasets[0].field_dict

        for f_name in intersection_field_names:
            field_mapping[f_name] = field_overrides.get(
                f_name, default_field_dict[f_name]
            )

        return field_mapping

    def _get_examples(self) -> List[Example]:
        return list(self)

    def _reverse_field_name_mapping(self):
        if self._reverse_field_name_mapping_dict is None:
            self._reverse_field_name_mapping_dict = {
                mapped_field.name: orig_fname
                for orig_fname, mapped_field in self._field_mapping.items()
            }
        return self._reverse_field_name_mapping_dict

    def _update_override_fields(self) -> None:
        """
        Updates and finalizes all eager override fields.
        """
        eager_fields = {
            n: f for n, f in self._field_overrides.items() if not f.finalized and f.eager
        }

        if eager_fields:
            original_examples = chain(*self._datasets)
            for ex in original_examples:
                for original_field_name, override_field in eager_fields.items():
                    _, tokenized = ex[original_field_name]
                    override_field.update_vocab(tokenized)
            for eager_field in eager_fields.values():
                eager_field.finalize()

    def _map_example(self, example: Example) -> Example:
        """
        Transforms an example from a backing dataset into the format of the
        view, respecting field overrides.

        Parameters
        ----------
        example: Example
            Original Example to be mapped.
        Returns
        -------
        Example
            An example mapped to the format of this view.
        """
        new_example = Example()
        for original_field_name, mapped_field in self._field_mapping.items():
            new_example[mapped_field.name] = example[original_field_name]
        return new_example

    def _translate_index(self, index: int) -> Tuple[DatasetBase, int]:
        """
        For an index in the view, returns the backing Dataset it belongs to and
        the index of the example in that Dataset.

        Parameters
        ----------
        index: int
            The index to be translated.

        Returns
        -------
        (DatasetBase, int)
            The dataset that contains the indexed example and its index in that
            dataset.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range. Length: {len(self)}")

        if index < 0:
            # correct for negative indexing
            index %= len(self)

        # Use binary search to determine the index of the containing dataset
        dataset_index = bisect_right(self._cumulative_lengths, index)
        offset = self._cumulative_lengths[dataset_index - 1] if dataset_index > 0 else 0
        translated_index = index - offset
        dataset = self._datasets[dataset_index]
        return dataset, translated_index

    @staticmethod
    def _get_intersection_field_names(datasets: List[DatasetBase]) -> List[str]:
        field_dict = datasets[0].field_dict
        intersection_field_names = set(field_dict.keys())
        for ds in datasets[1:]:
            # Calculate the intersection of all field names
            intersection_field_names.intersection_update(ds.field_dict.keys())
        return list(intersection_field_names)


class DatasetIndexedView(DatasetBase):
    """
    View over a DatasetBase class.
    """

    def __init__(self, dataset: DatasetBase, indices: Sequence[int]):
        """
        Creates a view over the passed dataset.

        Parameters
        ----------
        dataset: DatasetBase
            The dataset the view will be created over.
        indices: Sequence[int]
            A sequence of ints that represent the indices of the examples in the
            dataset that will be contained in the view. Ordering and duplication
            will be respected.
        """
        if not isinstance(dataset, DatasetBase):
            err_msg = (
                f"'dataset' parameter must be of type DatasetBase. "
                f"Passed type: {type(dataset).__name__}"
            )
            raise TypeError(err_msg)

        self._dataset = dataset
        self._indices = indices
        super().__init__(dataset.fields)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, item):
        if isinstance(item, int):
            index = self._indices[item]
            return self._dataset[index]

        if isinstance(item, slice):
            new_indices = self._indices[item]
            return DatasetIndexedView(self._dataset, new_indices)

        else:
            new_indices = [self._indices[i] for i in item]
            return DatasetIndexedView(self._dataset, new_indices)

    def __iter__(self):
        for i in self._indices:
            yield self._dataset[i]

    def _get_examples(self) -> List[Example]:
        return list(self)


class DatasetSlicedView(DatasetBase):
    """
    View over a DatasetBase class.
    """

    def __init__(self, dataset: DatasetBase, s: slice):
        """
        Creates a view over the passed dataset.

        Parameters
        ----------
        dataset: DatasetBase
            The dataset the view will be created over.
        s: slice
            A slice indexing the wanted examples.
        """
        if not isinstance(dataset, DatasetBase):
            err_msg = (
                f"'dataset' parameter must be of type DatasetBase. "
                f"Passed type: {type(dataset).__name__}"
            )
            raise TypeError(err_msg)

        if not isinstance(s, slice):
            err_msg = (
                f"'s' parameter must be of type slice. "
                f"Passed type: {type(s).__name__}"
            )
            raise TypeError(err_msg)

        self._dataset = dataset
        start, stop, step = s.indices(len(dataset))
        self._slice = slice(start, stop, step)
        self._len = self._calculate_length()
        super().__init__(dataset.fields)

    def _calculate_length(self) -> int:
        """
        Calculates the number of examples in this view.

        Returns
        -------
        int:
            The number of examples in this view.
        """
        start, stop, step = self._slice.start, self._slice.stop, self._slice.step
        if step < 0:
            start, stop = stop, start
            step *= -1

        return ceil(max(stop - start, 0) / step)

    def __len__(self):
        return self._len

    def __iter__(self):
        start, stop, step = self._slice.start, self._slice.stop, self._slice.step
        for i in range(start, stop, step):
            yield self._dataset[i]

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self):
                err_msg = f"Index {item} out of bounds. Length: {len(self)}."
                raise IndexError(err_msg)

            if item < 0:
                item %= len(self)

            index = self._slice.start + item * self._slice.step
            return self._dataset[index]

        else:
            return create_view(self, item)

    def _get_examples(self) -> List[Example]:
        return list(self)


def check_split_ratio(split_ratio):
    """
    Checks that the split ratio argument is not malformed and if not transforms
    it to a tuple of (train_size, valid_size, test_size) and normalizes it if
    necessary so that all elements sum to 1.

    (See Dataset.split docs for more info).

    Parameters
    ----------
    split_ratio : (float | list[float] | tuple[float])
        The split_ratio should either be a float in the interval (0.0, 1.0)
        (size of train) or a list / tuple of floats of length 2 (or 3) that
        are all larger than 0 and that represent the relative sizes of train,
        (val), test splits.
        If given as a list / tuple, the relative sizes don't have to sum  up
        to 1.0 (they are normalized automatically).

    Returns
    -------
    tuple[float]
        A tuple of (train_size, valid_size, test_size) whose elements sum
        to 1.0.

    Raises
    ------
    ValueError
        If the ratio doesn't obey any of the expected formats described above.
    """

    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        if not (0.0 < split_ratio < 1.0):
            raise ValueError(f"Split ratio {split_ratio} not between 0 and 1")

        train_ratio = split_ratio
        val_ratio = None
        test_ratio = 1.0 - split_ratio
    elif isinstance(split_ratio, list) or isinstance(split_ratio, tuple):
        # A list/tuple of relative ratios is provided
        split_ratio = tuple(split_ratio)
        length = len(split_ratio)

        if length not in {2, 3}:
            raise ValueError(
                f"Split ratio list/tuple should be of length 2 or 3, got {length}"
            )

        for i, ratio in enumerate(split_ratio):
            if float(ratio) <= 0.0:
                raise ValueError(
                    f"Elements of ratio tuple/list must be > 0.0 "
                    f"(got value {ratio} at index {i})."
                )

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.0:
            split_ratio = tuple(float(ratio) / ratio_sum for ratio in split_ratio)

        train_ratio = split_ratio[0]
        if length == 2:
            val_ratio = None
            test_ratio = split_ratio[1]
        else:
            val_ratio = split_ratio[1]
            test_ratio = split_ratio[2]
    else:
        raise ValueError(
            f"Split ratio must be a float, a list or a tuple, but got {type(split_ratio)}"
        )

    return train_ratio, val_ratio, test_ratio


def rationed_split(examples, train_ratio, val_ratio, test_ratio, shuffle):
    """
    Splits a list of examples according to the given ratios and returns the
    splits as a tuple of lists (train_examples, valid_examples, test_examples).

    The list can also be randomly shuffled before splitting.

    Parameters
    ----------
    examples : list
        A list of examples that is to be split according to the ratios.
    train_ratio : float
        The fraction of examples that should be put into the train split.
    val_ratio : float
        The fraction of examples that should be put into the valid split.
    test_ratio : float
        The fraction of examples that should be put into the test split.
    shuffle : bool
        Whether to shuffle the list before splitting.

    Returns
    -------
    tuple
        The train, valid and test splits, each as a list of examples.

    Raises
    ------
    ValueError
        If the given split ratio is wrong in the sense that it would result
        with at least one empty split.
    """

    # Create a random permutation of examples, then split them
    # by ratio x length slices for each of the train/test/dev? splits
    N = len(examples)

    indices = list(range(N))
    if shuffle:
        random.shuffle(indices)

    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if val_ratio is None:
        if train_len == 0 or (N - train_len) == 0:
            raise ValueError("Bad ratio: both splits should have at least 1 element.")

        indices_tuple = (indices[:train_len], [], indices[train_len:])
    else:
        test_len = int(round(test_ratio * N))
        val_len = N - train_len - test_len

        if train_len * test_len * val_len == 0:
            raise ValueError("Bad ratio: all splits should have at least 1 element.")

        indices_tuple = (
            indices[:train_len],  # Train
            indices[train_len : train_len + val_len],  # Validation
            indices[train_len + val_len :],  # Test
        )

    # Create a tuple of 3 lists, the middle of which is empty if only the
    # train and test ratios were provided
    data = tuple([examples[idx] for idx in indices] for indices in indices_tuple)

    return data


def stratified_split(
    examples, train_ratio, val_ratio, test_ratio, strata_field_name, shuffle
):
    """
    Performs a stratified split on a list of examples according to the given
    ratios and the given strata field.

    Returns the splits as a tuple of lists (train_examples, valid_examples,
    test_examples).

    The list can also be randomly shuffled before splitting.

    Parameters
    ----------
    examples : list
        A list of examples that is to be split according to the ratios.
    train_ratio : float
        The fraction of examples that should be put into the train split.
    val_ratio : float
        The fraction of examples that should be put into the valid split.
    test_ratio : float
        The fraction of examples that should be put into the test split.
    strata_field_name : str
        Name of the field that the examples should be stratified over.
        The values of the strata field have to be hashable.
        Default is 'label' for the conventional label field.
    shuffle : bool
        Whether to shuffle the list before splitting.

    Returns
    -------
    tuple
        The stratified train, valid and test splits, each as a list of
        examples.
    """

    # group the examples by the strata_field
    strata = itertools.groupby(examples, key=lambda ex: ex[strata_field_name])
    strata = (list(group) for _, group in strata)

    train_split, val_split, test_split = [], [], []
    for group in strata:
        # Split each group of examples according to the ratios given
        group_train_split, group_val_split, group_test_split = rationed_split(
            group, train_ratio, val_ratio, test_ratio, shuffle
        )

        # add the group splits to total splits
        train_split += group_train_split
        val_split += group_val_split
        test_split += group_test_split

    # now, for each concrete label value (stratum) - as well as for the whole
    # list of examples - the ratios are preserved
    return train_split, val_split, test_split


def concat(
    datasets: List[DatasetBase],
    field_overrides: Optional[Union[Dict[str, Field], List[Field]]] = None,
) -> DatasetBase:
    """
    Concatenate datasets in the passed order.

    Two or more datasets can be concatenated. New fields can be provided as
    'field_overrides' that will be updated with all examples.

    Parameters
    ----------
        datasets: List[DatasetBase]
            A list datasets to be concatenated.
        field_overrides: Union[Dict[str, Field], List[Field]]
            A dict or list containing fields that will be used to override
            existing fields. Can be either a dict mapping old field names to new
            ones, or a list, in which case the field with the same name will be
            overridden. The overridden field will not be present in the
            concatenated view. The override field (if eager) will be updated
            with all examples from the concatenation.

    Returns
    -------
    DatasetBase
        A view of the concatenated Datasets.
    """
    return DatasetConcatView(datasets, field_overrides)


def create_view(dataset: DatasetBase, i: Union[Sequence[int], slice]) -> DatasetBase:
    """
    Creates a view that is appropriate for the passed indexing method.

    Parameters
    ----------
    dataset: DatasetBase
        The dataset the view will be created on.
    i: Union[Sequence[int], slice]
        The indices contained in the view.

    Returns
    -------
        A view on the passed dataset.
    """
    if isinstance(i, slice):
        return DatasetSlicedView(dataset, i)
    else:
        return DatasetIndexedView(dataset, i)
