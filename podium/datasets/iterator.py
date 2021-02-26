"""
Module contains classes for iterating over datasets.
"""
import math
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from random import Random
from typing import Callable
from typing import Iterator as PythonIterator
from typing import List, NamedTuple, Tuple

import numpy as np

from podium.datasets.dataset import Dataset, DatasetBase
from podium.datasets.hierarhical_dataset import HierarchicalDataset


class Batch(dict):
    def __iter__(self):
        yield from self.values()

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class IteratorBase(ABC):
    """
    Abstract base class for all Iterators in Podium.
    """

    def __call__(
        self, dataset: DatasetBase
    ) -> PythonIterator[Tuple[NamedTuple, NamedTuple]]:
        """
        Sets the dataset for this Iterator and returns an iterable over the
        batches of that dataset. Same as calling iterator.set_dataset() followed
        by iter(iterator)

        Parameters
        ----------
        dataset: Dataset
            Dataset to iterate over.

        Returns
        -------
            Iterable over batches in the Dataset.
        """
        self.set_dataset(dataset)
        return iter(self)

    @abstractmethod
    def set_dataset(self, dataset: DatasetBase) -> None:
        """
        Sets the dataset for this Iterator to iterate over. Resets the epoch
        count.

        Parameters
        ----------
        dataset: DatasetBase
            Dataset to iterate over.
        """
        pass

    @abstractmethod
    def __iter__(self) -> PythonIterator[Tuple[NamedTuple, NamedTuple]]:
        """
        Returns an iterator object that knows how to iterate over the given
        dataset. The iterator yields tuples in the form (input_batch,
        target_batch). The input_batch and target_batch objects have attributes
        that correspond to the names of input fields and target fields
        (respectively) of the dataset.

        Returns
        -------
        iter
            Iterator that iterates over batches of examples in the dataset.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of batches this iterator provides in one epoch.

        Returns
        -------
        int
            Number of batches s provided in one epoch.
        """
        pass


class Iterator(IteratorBase):
    """
    An iterator that batches data from a dataset after numericalization.
    """

    def __init__(
        self,
        dataset=None,
        batch_size=32,
        sort_key=None,
        shuffle=True,
        seed=1,
        matrix_class=np.array,
        internal_random_state=None,
    ):
        """
        Creates an iterator for the given dataset and batch size.

        Parameters
        ----------
        dataset : DatasetBase
            The dataset whose examples the iterator will iterate over.
        batch_size : int
            The size of the batches that the iterator will return. If the
            number of examples in the dataset is not a multiple of
            batch_size the last returned batch will be smaller
            (dataset_len MOD batch_size).
        sort_key : callable
            A callable object used to sort the instances in a batch.
            If None, the batch instances won't be sorted.
            Default is None.
        shuffle : bool
            A flag denoting whether the examples should be shuffled before
            each iteration.
            If sort_key is not None, this flag being True may not have any
            effect since the dataset will always be sorted after being
            shuffled (the only difference shuffling can make is in the
            order of elements with the same value of sort_key)..
            Default is False.
        seed : int
            The seed that the iterator's internal random state will be
            initialized with. Useful when we want repeatable random shuffling.
            Only relevant if shuffle is True. If shuffle is True and
            internal_random_state is None, then this must not be None,
            otherwise a ValueError is raised.
            Default is 1.
        internal_random_state : tuple
            The random state that the iterator will be initialized with.
            Useful when we want to stop iteration and later continue where
            we left off.
            If None, the iterator will create its own random state from the
            given seed, that can later be obtained if we want to store it.
            Only relevant if shuffle is True. If shuffle is True and seed is
            None, then this must not be None, otherwise a ValueError is
            raised.
            Default is None.

        Raises
        ------
        ValueError
            If shuffle is True and both seed and internal_random_state are
            None.
        """

        self._batch_size = batch_size

        self._shuffle = shuffle

        self._sort_key = sort_key

        self._epoch = 0
        self._iterations = 0
        self._matrix_class = matrix_class

        # set of fieldnames for which numericalization format warnings were issued
        # used to avoid spamming warnings between iterations
        self._numericalization_format_warned_fieldnames = set()

        if dataset is not None:
            self.set_dataset(dataset)

        else:
            self._dataset = None

        if self._shuffle:
            if seed is None and internal_random_state is None:
                raise ValueError(
                    "If shuffle==True, either seed or "
                    "internal_random_state have to be != None."
                )

            self._shuffler = Random(seed)

            if internal_random_state is not None:
                self._shuffler.setstate(internal_random_state)
        else:
            self._shuffler = None

    @property
    def epoch(self) -> int:
        """
        The current epoch of the Iterator.
        """
        return self._epoch

    @property
    def iterations(self) -> int:
        """
        Number of batches returned for the current epoch.
        """
        return self._iterations

    @property
    def matrix_class(self):
        """
        The class constructor of the batch matrix.
        """
        return self._matrix_class

    @property
    def batch_size(self):
        """
        The batch size of the iterator.
        """
        return self._batch_size

    @property
    def sort_key(self):
        return self._sort_key

    def reset(self):
        """
        Reset the epoch and iteration counter of the Iterator.
        """
        self._epoch = 0
        self._iterations = 0

    def set_dataset(self, dataset: DatasetBase) -> None:
        """
        Sets the dataset for this Iterator to iterate over. Resets the epoch
        count.

        Parameters
        ----------
        dataset: DatasetBase
            Dataset to iterate over.
        """
        self.reset()

        self._dataset = dataset

    def __setstate__(self, state):
        self.__dict__ = state
        if self._shuffle:
            # Restore the random state to the one prior to start
            # of last epoch so we can rewind to the correct batch
            self.set_internal_random_state(self._shuffler_state)

    def __len__(self) -> int:
        """
        Returns the number of batches this iterator provides in one epoch.

        Returns
        -------
        int
            Number of batches s provided in one epoch.
        """

        return math.ceil(len(self._dataset) / self.batch_size)

    def __iter__(self) -> PythonIterator[Tuple[NamedTuple, NamedTuple]]:
        """
        Returns an iterator object that knows how to iterate over the given
        dataset. The iterator yields tuples in the form (input_batch,
        target_batch). The input_batch and target_batch objects have attributes
        that correspond to the names of input fields and target fields
        (respectively) of the dataset. The values of those attributes are numpy
        matrices, whose rows are the numericalized values of that field in the
        examples that are in the batch. Rows of sequential fields (that are of
        variable length) are all padded to a common length. The common length is
        either the fixed_length attribute of the field or, if that is not given,
        the maximum length of all examples in the batch.

        Returns
        -------
        iter
            Iterator that iterates over batches of examples in the dataset.
        """
        indices = list(range(len(self._dataset)))

        if self._shuffle:
            # Cache state prior to shuffle so we can use it when unpickling
            self._shuffler_state = self.get_internal_random_state()
            self._shuffler.shuffle(indices)

        data = self._dataset[indices]

        # If iteration was stopped, continue where we left off
        start = self.iterations * self.batch_size

        for i in range(start, len(data), self.batch_size):
            batch_instances = data[i : i + self.batch_size]

            if self._sort_key is not None:
                batch_instances = batch_instances.sorted(key=self._sort_key)

            self._iterations += 1
            yield self._create_batch(batch_instances)

        # prepare for new epoch
        self._iterations = 0
        self._epoch += 1

    def _create_batch(self, dataset: DatasetBase) -> Tuple[NamedTuple, NamedTuple]:

        examples = dataset.examples

        # dicts that will be used to store the input and target batch
        input_batch = Batch()
        target_batch = Batch()

        for field in dataset.fields:
            numericalizations = []

            for example in examples:
                numericalization = field.get_numericalization_for_example(example)
                numericalizations.append(numericalization)

            # casting to matrix can only be attempted if all values are either
            # None or np.ndarray
            possible_cast_to_matrix = all(
                x is None or isinstance(x, (np.ndarray, int, float))
                for x in numericalizations
            )

            if (
                not possible_cast_to_matrix
                and not field._disable_batch_matrix
                and field.name not in self._numericalization_format_warned_fieldnames
            ):
                warnings.warn(
                    f"The batch for Field '{field.name}' can't be cast to "
                    f"matrix but `disable_batch_matrix` is set to False."
                )
                self._numericalization_format_warned_fieldnames.add(field.name)

            if (
                len(numericalizations) > 0
                and not field._disable_batch_matrix
                and possible_cast_to_matrix
            ):
                batch = Iterator._arrays_to_matrix(
                    field, numericalizations, self.matrix_class
                )

            else:
                batch = numericalizations

            if field.include_lengths:
                # Include the length of each instance in the Field
                # along with the numericalization
                batch_lengths = self.matrix_class(
                    [len(instance) for instance in numericalizations]
                )
                batch = (batch, batch_lengths)

            if field.is_target:
                target_batch[field.name] = batch
            else:
                input_batch[field.name] = batch
        return input_batch, target_batch

    def get_internal_random_state(self):
        """
        Returns the internal random state of the iterator.

        Useful when we want to stop iteration and later continue where we left
        off. We can store the random state obtained with this method and later
        initialize another iterator with the same random state and continue
        iterating.

        Only to be called if shuffle is True, otherwise a RuntimeError is
        raised.

        Returns
        -------
        tuple
            The internal random state of the iterator.

        Raises
        ------
        RuntimeError
            If shuffle is False.
        """

        if not self._shuffle:
            raise RuntimeError(
                "Iterator with shuffle=False does not have an internal random state."
            )

        return self._shuffler.getstate()

    def set_internal_random_state(self, state):
        """
        Sets the internal random state of the iterator.

        Useful when we want to stop iteration and later continue where we left
        off. We can take the random state previously obtained from another
        iterator to initialize this iterator with the same state and continue
        iterating where the previous iterator stopped.

        Only to be called if shuffle is True, otherwise a RuntimeError is
        raised.

        Raises
        ------
        RuntimeError
            If shuffle is False.
        """

        if not self._shuffle:
            raise RuntimeError(
                "Iterator with shuffle=False does not have an internal random state."
            )

        self._shuffler.setstate(state)

    @staticmethod
    def _arrays_to_matrix(
        field, arrays: List[np.ndarray], matrix_class: Callable
    ) -> np.ndarray:
        pad_length = Iterator._get_pad_length(field, arrays)
        padded_arrays = [field._pad_to_length(a, pad_length) for a in arrays]
        return matrix_class(padded_arrays)

    @staticmethod
    def _get_pad_length(field, numericalizations) -> int:
        # the fixed_length attribute of Field has priority over the max length
        # of all the examples in the batch
        if field._fixed_length is not None:
            return field._fixed_length

        # if fixed_length is None, then return the maximum length of all the
        # examples in the batch
        def numericalization_length(n):
            if n is None or isinstance(n, (int, float)):
                return 1
            else:
                return len(n)

        return max(map(numericalization_length, numericalizations))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({{"
            f"batch_size: {self._batch_size}, "
            f"sort_key: {self._sort_key}, "
            f"shuffle: {self._shuffle}}})"
        )


class SingleBatchIterator(Iterator):
    """
    Iterator that creates one batch per epoch containing all examples in the
    dataset.
    """

    def __init__(self, dataset: DatasetBase = None, shuffle=True):
        """
        Creates an Iterator that creates one batch per epoch containing all
        examples in the dataset.

        Parameters
        ----------
        dataset : DatasetBase
            The dataset whose examples the iterator will iterate over.

        shuffle : bool
            A flag denoting whether the examples should be shuffled before
            each iteration.
            If sort_key is not None, this flag being True may not have any
            effect since the dataset will always be sorted after being
            shuffled (the only difference shuffling can make is in the
            order of elements with the same value of sort_key)..
            Default is False.
        """
        super().__init__(dataset=dataset, batch_size=len(dataset), shuffle=shuffle)

    def set_dataset(self, dataset: DatasetBase) -> None:
        super().set_dataset(dataset)
        self._batch_size = len(dataset)

    def __len__(self) -> int:
        return 1


class BucketIterator(Iterator):
    """
    Creates a bucket iterator that uses a look-ahead heuristic to try and batch
    examples in a way that minimizes the amount of necessary padding.

    It creates a bucket of size N x batch_size, and sorts that bucket before
    splitting it into batches, so there is less padding necessary.
    """

    def __init__(
        self,
        dataset=None,
        batch_size=32,
        sort_key=None,
        shuffle=True,
        seed=1,
        matrix_class=np.array,
        internal_random_state=None,
        look_ahead_multiplier=100,
        bucket_sort_key=None,
    ):
        """
        Creates a BucketIterator with the given bucket sort key and look-ahead
        multiplier (how many batch_sizes to look ahead when sorting examples for
        batches).

        Parameters
        ----------
        look_ahead_multiplier : int
            Number that denotes how many times the look-ahead bucket is larger
            than the batch_size.
            If look_ahead_multiplier == 1, then BucketIterator behaves like a
            normal iterator except with sorting within the batches.
            If look_ahead_multiplier >= (num_examples / batch_size), then
            BucketIterator behaves like a normal iterator that sorts the
            whole dataset.
            Default is 100.
        bucket_sort_key : callable
            The callable object used to sort the examples in the bucket that
            is to be batched.
            If bucket_sort_key is None, then sort_key must not be None,
            otherwise a ValueError is raised.
            Default is None.

        Raises
        ------
        ValueError
            If sort_key and bucket_sort_key are both None.
        """

        if sort_key is None and bucket_sort_key is None:
            raise ValueError(
                "For BucketIterator to work, either sort_key or "
                "bucket_sort_key must be != None."
            )

        super().__init__(
            dataset,
            batch_size,
            sort_key=sort_key,
            shuffle=shuffle,
            seed=seed,
            matrix_class=matrix_class,
            internal_random_state=internal_random_state,
        )

        self.bucket_sort_key = bucket_sort_key
        self.look_ahead_multiplier = look_ahead_multiplier

    def __iter__(self) -> PythonIterator[Tuple[NamedTuple, NamedTuple]]:
        step = self.batch_size * self.look_ahead_multiplier
        dataset = self._dataset

        # Determine the step where iteration was stopped for lookahead & within bucket
        lookahead_start = (
            self.iterations // self.look_ahead_multiplier * self.look_ahead_multiplier
        )
        batch_start = self.iterations % self.look_ahead_multiplier

        if self._sort_key is not None:
            dataset = dataset.sorted(key=self._sort_key)
        for i in range(lookahead_start, len(dataset), step):
            bucket = dataset[i : i + step]

            if self.bucket_sort_key is not None:
                bucket = bucket.sorted(key=self.bucket_sort_key)

            for j in range(batch_start, len(bucket), self.batch_size):
                batch_dataset = bucket[j : j + self.batch_size]
                input_batch, target_batch = self._create_batch(batch_dataset)

                yield input_batch, target_batch
                self._iterations += 1

        # prepare for new epoch
        self._iterations = 0
        self._epoch += 1

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({{"
            f"batch_size: {self._batch_size}, "
            f"sort_key: {self._sort_key}, "
            f"shuffle: {self._shuffle}, "
            f"look_ahead_multiplier: {self.look_ahead_multiplier},"
            f"bucket_sort_key: {self.bucket_sort_key}}})"
        )


class HierarchicalDatasetIterator(Iterator):
    """
    Iterator used to create batches for Hierarchical Datasets.

    It creates batches in the form of lists of matrices. In the batch namedtuple
    that gets returned, every attribute corresponds to a field in the dataset.
    For every field in the dataset, the namedtuple contains a list of matrices,
    where every matrix represents the context of an example in the batch. The
    rows of a matrix contain numericalized representations of the examples that
    make up the context of an example in the batch with the representation of
    the example itself being in the last row of its own context matrix.
    """

    def __init__(
        self,
        dataset=None,
        batch_size=32,
        sort_key=None,
        shuffle=False,
        seed=1,
        matrix_class=np.array,
        internal_random_state=None,
        context_max_length=None,
        context_max_depth=None,
    ):
        """
        Creates an iterator for the given dataset and batch size.

        Parameters
        ----------
        dataset : DatasetBase
            The dataset whose examples the iterator will iterate over.
        batch_size : int
            The size of the batches that the iterator will return. If the
            number of examples in the dataset is not a multiple of
            batch_size the last returned batch will be smaller
            (dataset_len MOD batch_size). Defaults to 32.
        sort_key : callable
            A callable object used to sort the dataset prior to batching. If
            None, the dataset won't be sorted.
            Default is None.
        shuffle : bool
            A flag denoting whether the examples should be shuffled before
            each iteration.
            If sort_key is not None, this flag being True may not have any
            effect since the dataset will always be sorted after being
            shuffled (the only difference shuffling can make is in the
            order of elements with the same value of sort_key)..
            Default is False.
        seed : int
            The seed that the iterator's internal random state will be
            initialized with. Useful when we want repeatable random shuffling.
            Only relevant if shuffle is True. If shuffle is True and
            internal_random_state is None, then this must not be None,
            otherwise a ValueError is raised.
            Default is 1.
        internal_random_state : tuple
            The random state that the iterator will be initialized with.
            Useful when we want to stop iteration and later continue where
            we left off.
            If None, the iterator will create its own random state from the
            given seed, that can later be obtained if we want to store it.
            Only relevant if shuffle is True. If shuffle is True and seed is
            None, then this must not be None, otherwise a ValueError is
            raised.
            Default is None.
        context_max_depth: int
            The maximum depth of the context retrieved for an example in the batch.
            While generating the context, the iterator will take 'context_max_depth'
            levels above the example and the root node of the last level, e.g. if 0 is
            passed, the context generated for an example will contain all examples in the
            level of the example in the batch and the root example of that level.
            If None, this depth limit will be ignored.
        context_max_length: int
            The maximum length of the context retrieved for an example in the batch. The
            number of rows in the generated context matrix will be (if needed) truncated
            to `context_max_length` - 1.
            If None, this length limit will be ignored.

        Raises
        ------
        ValueError
            If shuffle is True and both seed and internal_random_state are
            None.
        """

        if context_max_length is not None and context_max_length < 1:
            raise ValueError(
                "'context_max_length' must not be less than 1. "
                "If you don't want context, try flattening the dataset. "
                f"'context_max_length' : {context_max_length})"
            )

        if context_max_depth is not None and context_max_depth < 0:
            raise ValueError(
                "'context_max_depth' must not be negative. "
                f"'context_max_depth' : {context_max_length}"
            )

        self._context_max_depth = context_max_depth
        self._context_max_length = context_max_length

        super().__init__(
            dataset,
            batch_size,
            sort_key=sort_key,
            shuffle=shuffle,
            seed=seed,
            internal_random_state=internal_random_state,
        )

    def set_dataset(self, dataset: HierarchicalDataset) -> None:
        if not isinstance(dataset, HierarchicalDataset):
            err_msg = (
                f"HierarchicalIterator can only iterate over "
                f"HierarchicalDatasets. Passed dataset type: "
                f"{type(dataset).__name__}"
            )
            raise ValueError(err_msg)
        super().set_dataset(dataset)

    def _get_node_context(self, node):
        """
        Generates a list of examples that make up the context of the provided
        node, truncated to adhere to 'context_max_depth' and
        'context_max_length' limitations.

        Parameters
        ----------
        node : Node
            The Hierarchical dataset node the context should be retrieved for.

        Returns
        -------
        list(Example)
            A list of examples that make up the context of the provided node,
            truncated to adhere to 'context_max_depth' and 'context_max_length'
            limitations.
        """
        context_iterator = HierarchicalDataset._get_node_context(
            node, self._context_max_depth
        )
        context = list(context_iterator)

        if self._context_max_length is not None:
            # if context max size is defined, truncate it
            context = context[-self._context_max_length :]

        # add the example to the end of its own context
        context.append(node.example)

        return context

    def _nodes_to_batch(self, nodes):
        """
        Creates a batch from the passed nodes.

        Parameters
        ----------
        nodes : list(Node)
            Nodes that should be contained in the batch

        Returns
        -------
        (Batch, Batch)
            a tuple of two Batch instances, the input batch and target batch, containing
            the input and target fields of the batch respectively.
        """

        input_batch_dict, target_batch_dict = defaultdict(list), defaultdict(list)

        for node in nodes:
            # all examples that make up the current node's context
            node_context_examples = self._get_node_context(node)
            node_context_dataset = Dataset(node_context_examples, self._dataset.fields)
            input_sub_batch, target_sub_batch = super()._create_batch(
                node_context_dataset
            )

            for key in input_sub_batch.keys():
                value = getattr(input_sub_batch, key)
                input_batch_dict[key].append(value)

            for key in target_sub_batch.keys():
                value = getattr(target_sub_batch, key)
                target_batch_dict[key].append(value)

        input_batch = Batch(input_batch_dict)
        target_batch = Batch(target_batch_dict)

        return input_batch, target_batch

    def _data(self):
        """Generates a list of Nodes to be used in batch iteration.
        Returns
        -------
        list(Node)
            a list of Nodes
        """
        dataset_nodes = list(self._dataset._node_iterator())

        if self._shuffle:
            # shuffle the indices
            indices = list(range(len(self._dataset)))
            self._shuffler.shuffle(indices)

            # creates a new list of nodes
            dataset_nodes = [dataset_nodes[i] for i in indices]

        return dataset_nodes

    def __iter__(self) -> PythonIterator[Tuple[NamedTuple, NamedTuple]]:
        dataset_nodes = self._data()

        # If iteration was stopped, continue where we left off
        start = self.iterations * self.batch_size

        for i in range(start, len(dataset_nodes), self.batch_size):
            batch_nodes = dataset_nodes[i : i + self.batch_size]

            if self._sort_key is not None:
                batch_nodes = batch_nodes.sorted(
                    key=lambda node: self._sort_key(node.example)
                )

            yield self._nodes_to_batch(batch_nodes)
            self._iterations += 1

        # prepare for new epoch
        self._iterations = 0
        self._epoch += 1

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({{"
            f"batch_size: {self._batch_size}, "
            f"sort_key: {self._sort_key}, "
            f"shuffle: {self._shuffle}, "
            f"context_max_length: {self._context_max_length}, "
            f"context_max_depth: {self._context_max_depth}}})"
        )
