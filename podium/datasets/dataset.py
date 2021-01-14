"""
Module contains base classes for datasets.
"""
import copy
import itertools
import random
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np

from podium.field import Field, unpack_fields

from .example_factory import Example


FieldType = Optional[Union[Field, List[Field]]]


class DatasetBase(ABC):
    """
    Abstract base class for all datasets in Podium.
    """

    def __init__(self, fields: Union[Dict[str, FieldType], List[FieldType]]):
        self._fields = tuple(unpack_fields(fields))

    @property
    def fields(self) -> Tuple[Field]:
        """
        List containing all fields of this dataset.
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
        field_name : str
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
            # update _all_ non-eager fields
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

    def sorted(self, key: Callable[[Example], Any], reverse=False) -> "DatasetBase":
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

    def filtered(self, predicate: Callable[[Example], bool]) -> "DatasetBase":
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

    def shuffled(self) -> "DatasetBase":
        """
        Creates a new DatasetBase instance containing all Examples, but in
        shuffled order.

        Returns
        -------
        DatasetBase
            A new DatasetBase instance containing all Examples, but in shuffled
            order.
        """
        shuffled_indices = np.random.permutation(len(self))
        return self[shuffled_indices]

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
    def __getitem__(self, i: Iterable[int]) -> "DatasetBase":
        ...

    @abstractmethod
    def __getitem__(self, i: slice) -> "DatasetBase":
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
        self.sort_key = sort_key
        super().__init__(fields)

    def __getitem__(
        self, i: Union[int, Iterable[int], slice]
    ) -> Union["DatasetBase", Example]:
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
        i : int or slice or iterable
            Index used to index examples.

        Returns
        -------
        single example or Dataset
            If i is an int, a single example will be returned.
            If i is a slice or iterable, a copy of this dataset containing
            only the indexed examples will be returned.
        """

        return self.get(i)

    def get(self, i, deep_copy=False):
        """
        Returns an example or a new dataset containing the indexed examples.

        If indexed with an int, only the example at that position
        will be returned.
        If Indexed with a slice or iterable, all examples indexed by the object
        will be collected and a new dataset containing only those examples will be
        returned. The new dataset will contain copies of the old dataset's fields
        and will be identical to the original dataset, with the exception of the
        example number and ordering. See wiki for detailed examples.

        Example::

            # Indexing by a single integers returns a single example
            example = dataset.get(1)

            # Same as the first example, but returns a deep_copy of the example
            example_copy = dataset.get(1, deep_copy=True)

            # Multi-indexing returns a new dataset containing the indexed examples
            s = slice(1, 10)
            new_dataset = dataset.get(s)

            new_dataset_copy = dataset.get(s, deep_copy=True)

        Parameters
        ----------
        i : int or slice or iterable
            Index used to index examples.

        deep_copy: bool
            If true, the returned dataset will contain deep-copies of this
            dataset's examples and fields.
            If false, existing examples and fields will be reused.

        Returns
        -------
        single example or Dataset
            If i is an int, a single example will be returned.
            If i is a slice or iterable, a dataset containing
            only the indexed examples will be returned.
        """

        if isinstance(i, slice):
            return self._dataset_copy_with_examples(self.examples[i], deep_copy=deep_copy)

        elif isinstance(i, int):
            example = self.examples[i]
            return copy.deepcopy(example) if deep_copy else example

        else:
            # Numpy style multi-indexing
            indexed_examples = [self.examples[index] for index in i]
            return self._dataset_copy_with_examples(indexed_examples, deep_copy=deep_copy)

    def __len__(self) -> int:
        """
        Returns the number of examples in the dataset.

        Returns
        -------
        int
            The number of examples in the dataset.
        """
        return len(self._examples)

    def _get_examples(self) -> List[Example]:
        return self._examples

    def __iter__(self):
        """
        Iterates over all examples in the dataset in order.

        Yields
        ------
        example
            Yields examples in the dataset.
        """
        yield from self._examples

    def filter(self, predicate, inplace=False):
        """
        Method filters examples with given predicate.

        Parameters
        ----------
        predicate : callable
            predicate should be a callable that accepts example as input and returns
            true if the example shouldn't be filtered, otherwise returns false
        inplace : bool, default False
            if True, do operation inplace and return None
        """
        filtered_examples = [ex for ex in self.examples if predicate(ex)]

        if inplace:
            self._examples = filtered_examples
            return
        else:
            return Dataset(
                examples=filtered_examples, fields=self.fields, sort_key=self.sort_key
            )

    def filtered(self, predicate: Callable[[Example], bool]) -> "DatasetBase":
        return self.filter(predicate, inplace=False)

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
            Dataset(example_list, self.fields, sort_key=self.sort_key)
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

    def _dataset_copy_with_examples(
        self, examples: list, deep_copy: bool = False
    ) -> "Dataset":
        """
        Creates a new dataset with the same fields and sort_key. The new dataset
        contains only the fields passed to this function.Fields are deep-copied
        into the new dataset, but examples are used as-is.

        Parameters
        ----------
        examples
            examples to be kept in the copy of the dataset.

        deep_copy
            Whether to deep-copy the examples nad fields of this dataset.
            if False, existing fields and examples will be reused.

        Returns
        -------
        Dataset
            a copy of this dataset containing only the passed examples.
        """
        # Deep-copy if needed
        examples = copy.deepcopy(examples) if deep_copy else examples
        fields = copy.deepcopy(self.fields) if deep_copy else self.fields

        return Dataset(examples, fields, self.sort_key)

    def shuffle_examples(self, random_state=None):
        """
        Shuffles the examples in this dataset.

        Parameters
        ----------
        random_state : int
            The random seed used for shuffling.
        """

        if random_state is not None:
            random.seed(random_state)

        random.shuffle(self.examples)


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
