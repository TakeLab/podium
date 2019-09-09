""""Module contains classes for iterating over datasets."""
import math
import logging

from random import Random
from collections import namedtuple
import numpy as np

from takepod.datasets.dataset import Dataset
from takepod.datasets.hierarhical_dataset import HierarchicalDataset

_LOGGER = logging.getLogger(__name__)


class Iterator:
    """An iterator that batches data from a dataset after numericalization.

    Attributes
    ----------
    epoch : int
        The number of epochs elapsed up to this point.
    iterations : int
        The number of iterations elapsed in the current epoch.
    """

    def __init__(self,
                 dataset=None,
                 batch_size=32,
                 batch_to_matrix=True,
                 sort_key=None,
                 shuffle=False,
                 seed=1,
                 internal_random_state=None):
        """ Creates an iterator for the given dataset and batch size.

        Parameters
        ----------
        dataset : Dataset
            The dataset whose examples the iterator will iterate over.
        batch_size : int
            The size of the batches that the iterator will return. If the
            number of examples in the dataset is not a multiple of
            batch_size the last returned batch will be smaller
            (dataset_len MOD batch_size).
        batch_to_matrix : bool
            A flag denoting whether the vectors for a field in a batch should be
            returned as a list of numpy vectors or a matrix where each row is a padded
            vector
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

        Raises
        ------
        ValueError
            If shuffle is True and both seed and internal_random_state are
            None.
        """

        self.batch_size = batch_size
        self.batch_to_matrix = batch_to_matrix

        self.shuffle = shuffle

        self.sort_key = sort_key

        self.epoch = 0
        self.iterations = 0

        if dataset is not None:
            self.set_dataset(dataset)

        else:
            self._dataset = None

        if self.shuffle:
            if seed is None and internal_random_state is None:
                error_msg = "If shuffle==True, either seed or " \
                            "internal_random_state have to be != None."
                _LOGGER.error(error_msg)
                raise ValueError(error_msg)

            self.shuffler = Random(seed)

            if internal_random_state is not None:
                self.shuffler.setstate(internal_random_state)
        else:
            self.shuffler = None

    def set_dataset(self, dataset: Dataset):
        """Sets the dataset for this Iterator to iterate over.
        Resets the epoch count.

        Parameters
        ----------
        dataset: Dataset
            Dataset to iterate over.
        """
        self.epoch = 0
        self.iterations = 0

        self.input_batch_class = namedtuple(
            "InputBatch",
            [field.name for field in dataset.fields if not field.is_target]
        )

        self.target_batch_class = namedtuple(
            "TargetBatch",
            [field.name for field in dataset.fields if field.is_target]
        )

        self._dataset = dataset

    def __len__(self):
        """ Returns the number of batches this iterator provides in one epoch.

        Returns
        -------
        int
            Number of batche s provided in one epoch.
        """

        return math.ceil(len(self._dataset) / self.batch_size)

    def __iter__(self):
        """ Returns an iterator object that knows how to iterate over the
        given dataset.
        The iterator yields tuples in the form (input_batch, target_batch).
        The input_batch and target_batch objects have attributes that
        correspond to the names of input fields and target fields
        (respectively) of the dataset.
        The values of those attributes are numpy matrices, whose rows are the
        numericalized values of that field in the examples that are in the
        batch.
        Rows of sequential fields (that are of variable length) are all padded
        to a common length. The common length is either the fixed_length
        attribute of the field or, if that is not given, the maximum length
        of all examples in the batch.

        Returns
        -------
        iter
            Iterator that iterates over batches of examples in the dataset.
        """

        data = self._data()
        for i in range(0, len(data), self.batch_size):
            batch_examples = data[i: i + self.batch_size]
            input_batch, target_batch = self._create_batch(batch_examples)

            yield input_batch, target_batch
            self.iterations += 1

        # prepare for new epoch
        self.iterations = 0
        self.epoch += 1

    def get_internal_random_state(self):
        """ Returns the internal random state of the iterator.

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

        if not self.shuffle:
            error_msg = "Iterator with shuffle=False does not have " \
                        "an internal random state."
            _LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        return self.shuffler.getstate()

    def set_internal_random_state(self, state):
        """ Sets the internal random state of the iterator.

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

        if not self.shuffle:
            error_msg = "Iterator with shuffle=False does not have " \
                        "an internal random state."
            _LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        self.shuffler.setstate(state)

    def _create_batch(self, examples):

        if self.batch_to_matrix:
            return self._create_matrix_batch(examples)

        else:
            return self._create_list_batch(examples)

    def _create_matrix_batch(self, examples):

        # dicts that will be used to create the InputBatch and TargetBatch
        # objects
        input_batch_dict, target_batch_dict = {}, {}

        for field in self._dataset.fields:
            # the length to which all the rows are padded (or truncated)
            pad_length = Iterator._get_pad_length(field, examples)

            # the last batch can have < batch_size examples
            n_rows = min(self.batch_size, len(examples))

            # empty matrix to be filled with numericalized fields
            matrix = None  # np.empty(shape=(n_rows, pad_length))

            # non-sequential fields all have length = 1, no padding necessary
            should_pad = True if field.sequential else False

            for i, example in enumerate(examples):

                # Get cached value
                row = field.get_numericalization_for_example(example)

                if matrix is None:
                    # Create matrix of the correct dtype
                    matrix = np.empty(shape=(n_rows, pad_length), dtype=row.dtype)

                if should_pad:
                    row = field.pad_to_length(row, pad_length)

                # set the matrix row to the numericalized, padded array
                matrix[i] = row

            if field.is_target:
                target_batch_dict[field.name] = matrix
            else:
                input_batch_dict[field.name] = matrix

        input_batch = self.input_batch_class(**input_batch_dict)
        target_batch = self.target_batch_class(**target_batch_dict)

        return input_batch, target_batch

    def _create_list_batch(self, examples):
        # dicts that will be used to create the InputBatch and TargetBatch
        # objects
        input_batch_dict, target_batch_dict = {}, {}
        for field in self._dataset.fields:

            vectors = [field.get_numericalization_for_example(ex)
                       for ex
                       in examples]

            if field.is_target:
                target_batch_dict[field.name] = vectors

            else:
                input_batch_dict[field.name] = vectors

        input_batch = self.input_batch_class(**input_batch_dict)
        target_batch = self.target_batch_class(**target_batch_dict)

        return input_batch, target_batch

    @staticmethod
    def _get_pad_length(field, examples):
        if not field.sequential:
            return 1

        # the fixed_length attribute of Field has priority over the max length
        # of all the examples in the batch
        if field.fixed_length is not None:
            return field.fixed_length

        # if fixed_length is None, then return the maximum length of all the
        # examples in the batch
        def length_of_field(example):
            _, tokens = getattr(example, field.name)
            return len(tokens)

        return max(map(length_of_field, examples))

    def _data(self):
        """Method returns the copy of examples in the dataset.

        The examples are shuffled if shuffle is True and sorted if sort_key
        is not None. Sorting happens after shuffling so the shuffle flag may
        have no effect if sort_key is not None (the only difference it can
        make is in the order of elements with the same value of sort_key).
        """

        if self.shuffle:
            # shuffle the indices
            indices = list(range(len(self._dataset)))
            self.shuffler.shuffle(indices)

            # creates a new list
            xs = [self._dataset[i] for i in indices]
        else:
            # copies the list
            xs = self._dataset.examples[:]

        # sorting the newly created list
        if self.sort_key is not None:
            xs.sort(key=self.sort_key)

        return xs


class SingleBatchIterator(Iterator):
    """ Iterator that creates one batch per epoch
    containing all examples in the dataset."""

    def __init__(
            self,
            dataset: Dataset = None,
            batch_to_matrix: bool = True):
        """Creates an Iterator that creates one batch per epoch
        containing all examples in the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset whose examples the iterator will iterate over.

        batch_to_matrix : bool
            A flag denoting whether the vectors for a field in a batch should be
            returned as a list of numpy vectors or a matrix where each row is a padded
            vector.
        """
        super().__init__(dataset=dataset,
                         batch_to_matrix=batch_to_matrix)

    def set_dataset(self, dataset: Dataset):
        super().set_dataset(dataset)
        self.batch_size = len(dataset)

    def __len__(self):
        return 1

    def __iter__(self):
        examples = self._data()
        input_batch, target_batch = self._create_batch(examples)
        yield input_batch, target_batch

        self.epoch += 1


class BucketIterator(Iterator):
    """ Creates a bucket iterator that uses a look-ahead heuristic to try and
    batch examples in a way that minimizes the amount of necessary padding.

    It creates a bucket of size N x batch_size, and sorts that bucket before
    splitting it into batches, so there is less padding necessary.
    """

    def __init__(
            self,
            dataset,
            batch_size,
            batch_to_matrix=True,
            sort_key=None,
            shuffle=True,
            seed=42,
            look_ahead_multiplier=100,
            bucket_sort_key=None,
    ):
        """Creates a BucketIterator with the given bucket sort key and
        look-ahead multiplier (how many batch_sizes to look ahead when
        sorting examples for batches).

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
            error_msg = "For BucketIterator to work, either sort_key or "\
                        "bucket_sort_key must be != None."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        super().__init__(dataset,
                         batch_size,
                         batch_to_matrix=batch_to_matrix,
                         sort_key=sort_key,
                         shuffle=shuffle,
                         seed=seed)

        self.bucket_sort_key = bucket_sort_key
        self.look_ahead_multiplier = look_ahead_multiplier

    def __iter__(self):
        """ Returns an iterator object that knows how to iterate over the
        batches of the given dataset.

        Returns
        -------
        iter
            Iterator that iterates over batches of examples in the dataset.
        """
        data = self._data()
        step = self.batch_size * self.look_ahead_multiplier

        for i in range(0, len(data), step):
            if self.bucket_sort_key is not None:
                bucket = sorted(data[i: i + step], key=self.bucket_sort_key)
            else:
                # if bucket_sort_key is None, sort_key != None so the whole
                # dataset was already sorted with sort_key
                bucket = data[i: i + step]

            for j in range(0, len(bucket), self.batch_size):
                examples = bucket[j: j + self.batch_size]
                input_batch, target_batch = self._create_batch(examples)

                yield input_batch, target_batch
                self.iterations += 1

        # prepare for new epoch
        self.iterations = 0
        self.epoch += 1


class HierarchicalDatasetIterator(Iterator):
    """
    Iterator used to create batches for Hierarchical Datasets.

    It creates batches in the form of lists of matrices. In the batch namedtuple that gets
    returned, every attribute corresponds to a field in the dataset. For every field in
    the dataset, the namedtuple contains a list of matrices, where every matrix
    represents the context of an example in the batch. The rows of a matrix contain
    numericalized representations of the examples that make up the context of an example
    in the batch with the representation of the example itself being in the last row of
    its own context matrix.

    """

    def __init__(
            self,
            dataset,
            batch_size,
            sort_key=None,
            shuffle=False,
            seed=1,
            internal_random_state=None,
            context_max_length=None,
            context_max_depth=None,
    ):
        """ Creates an iterator for the given dataset and batch size.

        Parameters
        ----------
        dataset : Dataset
            The dataset whose examples the iterator will iterate over.
        batch_size : int
            The size of the batches that the iterator will return. If the
            number of examples in the dataset is not a multiple of
            batch_size the last returned batch will be smaller
            (dataset_len MOD batch_size).
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
            error_msg = "'context_max_length' must not be less than 1. "\
                        "If you don't want context, try flattening the dataset. "\
                        "'context_max_length' : {})".format(context_max_length)
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        if context_max_depth is not None and context_max_depth < 0:
            error_msg = "'context_max_depth' must not be negative. "\
                        "'context_max_depth' : {}".format(context_max_depth)
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        self._context_max_depth = context_max_depth
        self._context_max_size = context_max_length

        super().__init__(
            dataset,
            batch_size,
            sort_key=sort_key,
            shuffle=shuffle,
            seed=seed,
            internal_random_state=internal_random_state,
        )

    def _get_node_context(self, node):
        """Generates a list of examples that make up the context of the provided node,
        truncated to adhere to 'context_max_depth' and 'context_max_length' limitations.

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

        if self._context_max_size is not None:
            # if context max size is defined, truncate it
            context = context[-self._context_max_size:]

        # add the example to the end of its own context
        context.append(node.example)

        return context

    def _create_batch(self, nodes):
        """
        Creates a batch from the passed nodes.
        Parameters
        ----------
        nodes : list(Node)
            Nodes that should be contained in the batch

        Returns
        -------
        (namedtuple, namedtuple)
            a tuple of two namedtuples, input batch and target batch, containing the
            input and target fields of the batch respectively.

        """

        input_batch_dict, target_batch_dict = {}, {}

        for field in self._dataset.fields:
            # list of matrices containing numericalized contexts for the current field
            field_contextualized_example_matrices = []

            for node in nodes:

                # all examples that make up the current node's context
                node_context_examples = self._get_node_context(node)

                # the length to which all the rows are padded (or truncated)
                pad_length = Iterator._get_pad_length(field, node_context_examples)

                # empty matrix to be filled with numericalized fields
                n_rows = len(node_context_examples)
                matrix = np.empty(shape=(n_rows, pad_length))

                # non-sequential fields all have length = 1, no padding necessary
                should_pad = True if field.sequential else False

                for i, example in enumerate(node_context_examples):
                    # Get cached value
                    row = field.get_numericalization_for_example(example)

                    if should_pad:
                        row = field.pad_to_length(row, pad_length)

                    # set the matrix row to the numericalized, padded array
                    matrix[i] = row

                field_contextualized_example_matrices.append(matrix)

            if field.is_target:
                target_batch_dict[field.name] = field_contextualized_example_matrices

            else:
                input_batch_dict[field.name] = field_contextualized_example_matrices

        input_batch = self.input_batch_class(**input_batch_dict)
        target_batch = self.target_batch_class(**target_batch_dict)

        return input_batch, target_batch

    def _data(self):
        """Generates a list of Nodes to be used in batch iteration.
        Returns
        -------
        list(Node)
            a list of Nodes
        """
        dataset_nodes = list(self._dataset._node_iterator())

        if self.shuffle:
            # shuffle the indices
            indices = list(range(len(self._dataset)))
            self.shuffler.shuffle(indices)

            # creates a new list of nodes
            dataset_nodes = [dataset_nodes[i] for i in indices]

        if self.sort_key is not None:
            dataset_nodes.sort(key=lambda node: self.sort_key(node.example))

        return dataset_nodes
