from abc import ABC, abstractmethod
from typing import Union, Iterable, Iterator, List, Callable, Tuple, Dict, NamedTuple, Any

from podium.storage import Example, Field
from podium.datasets import SingleBatchIterator


class DatasetABC(ABC):
    # ================= Default methods =======================
    def __getitem__(
        self, i: Union[int, Iterable[int], slice]
    ) -> Union["DatasetABC", Example]:
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

    def __iter__(self) -> Iterator[Example]:
        """Iterates over all examples in the dataset in order.

        Yields
        ------
        Example
            Yields examples in the dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, field_name: str) -> Iterator[Tuple[Any, Any]]:
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
        for example in self:
            return getattr(example, field_name)

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
                        _, tokenized = getattr(example, field.name)
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

    @property
    def fields(self) -> List[Field]:
        """List containing all fields of this dataset."""
        return self._fields_list()

    @property
    def field_dict(self) -> Dict[str, Field]:
        return {f.name: f for f in self.fields}

    @property
    def examples(self) -> List[Example]:
        return self.get_example_list()

    # ================= Abstract methods =======================

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def _fields_list(self) -> List[Field]:
        """Returns a list of all Fields in this dataset.

        Returns
        -------
        list of Fields
            a list of all Fields in this Dataset
        """
        pass

    @abstractmethod
    def _get(self, i: Union[int, Iterable[int], slice]) -> Union["DatasetABC", Example]:
        pass

    @abstractmethod
    def get_example_list(self) -> List[Example]:
        pass

    @abstractmethod
    def filtered(self, predicate: Callable[[Example], bool]) -> "DatasetABC":
        """Method filters examples with given predicate and returns a new DatasetABC
        instance containing those examples.

                Parameters
                ----------
                predicate : callable
                    predicate should be a callable that accepts example as input and returns
                    true if the example shouldn't be filtered, otherwise returns false
        """
        pass
