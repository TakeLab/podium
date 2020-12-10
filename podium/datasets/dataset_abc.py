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

from podium.storage import Example, Field, unpack_fields


FieldType = Optional[Union[Field, List[Field]]]


class DatasetABC(ABC):
    def __init__(self, fields: Union[Dict[str, FieldType], List[FieldType]]):
        self._fields = tuple(unpack_fields(fields))

    # ==================== Properties =========================

    @property
    def fields(self) -> Tuple[Field]:
        """List containing all fields of this dataset."""
        return self._fields

    @property
    def field_dict(self) -> Dict[str, Field]:
        """Dictionary containing all field names mapping to their respective Fields."""
        return {f.name: f for f in self.fields}

    @property
    def examples(self) -> List[Example]:
        """List containing all Examples."""
        return self._get_examples()

    # ================= Default methods =======================

    def __iter__(self) -> Iterator[Example]:
        """Iterates over all examples in the dataset in order.

        Yields
        ------
        Example
            Yields examples in the dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, field: Union[str, Field]) -> Iterator[Tuple[Any, Any]]:
        """Returns an Iterator iterating over values of the field with the
        given name for every example in the dataset.

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

    def finalize_fields(self, *datasets: "DatasetABC") -> None:
        """
        Builds vocabularies of all the non-eager fields in the dataset,
        from the Dataset objects given as \\*args and then finalizes all the
        fields.

        Parameters
        ----------
        \\*datasets
            A variable number of DatasetABC objects from which to build the
            vocabularies for non-eager fields. If none provided, the
            vocabularies are built from this Dataset (self).
        """

        # if there are non-eager fields, we need to build their vocabularies
        fields_to_build = [f for f in self.fields if not f.eager and f.use_vocab]
        if fields_to_build:
            # there can be multiple datasets we want to iterate over
            data_sources = [ds for ds in datasets if isinstance(ds, DatasetABC)]

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
        """Creates an input and target batch containing the whole dataset.
        The format of the batch is the same as the batches returned by the

        Returns
        -------
        input_batch, target_batch
                Two objects containing the input and target batches over
                the whole dataset.
        """
        # Imported here because of circular import
        from podium.datasets import SingleBatchIterator

        return next(iter(SingleBatchIterator(self, shuffle=False)))

    def sorted(self, key: Callable[[Example], Any], reverse=False) -> "DatasetABC":
        """Creates a new DatasetABC instance in which all Examples are sorted according to
        the value returned by `key`.

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
        DatasetABC
            A new DatasetABC instance with sorted Examples.
        """

        def index_key(i):
            return key(self[i])

        indices = list(range(len(self)))
        indices.sort(key=index_key, reverse=reverse)
        return self[indices]

    def filtered(self, predicate: Callable[[Example], bool]) -> "DatasetABC":
        """Filters examples with given predicate and returns a new DatasetABC
        instance containing those examples.

        Parameters
        ----------
        predicate : callable
            predicate should be a callable that accepts example as input and returns
            true if the example shouldn't be filtered, otherwise returns false

        Returns
        -------
        DatasetABC
            A new DatasetABC instance containing only the Examples for which `predicate`
            returned True.
        """
        indices = [i for i, example in enumerate(self) if predicate(example)]
        return self[indices]

    def shuffled(self) -> "DatasetABC":
        """Creates a new DatasetABC instance containing all Examples, but in shuffled
        order.

        Returns
        -------
        DatasetABC
            A new DatasetABC instance containing all Examples, but in shuffled
            order.
        """
        shuffled_indices = np.random.permutation(len(self))
        return self[shuffled_indices]

    def __repr__(self):
        return f"{type(self).__name__}[Size: {len(self)}, Fields: {self.fields}]"

    # ================= Abstract methods =======================

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of examples in the dataset.

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
    def __getitem__(self, i: Iterable[int]) -> "DatasetABC":
        ...

    @abstractmethod
    def __getitem__(self, i: slice) -> "DatasetABC":
        """Returns an example or a new dataset containing the indexed examples.

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
        """Returns a list containing all examples of this dataset."""
        pass
