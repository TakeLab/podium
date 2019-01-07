""""Module contains classes for iterating over datasets."""
import math
from random import Random
import numpy as np

from collections import namedtuple


class Iterator:
    """An iterator that batches data from a dataset after numericalization.

    Attributes
    ----------
    epoch : int
        The number of epochs elapsed up to this point.
    iterations : int
        The number of iterations elapsed in the current epoch.
    """

    def __init__(self, dataset, batch_size, sort_key=None, shuffle=False,
                 seed=1, internal_random_state=None):
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

        Raises
        ------
        ValueError
            If shuffle is True and both seed and internal_random_state are
            None.
        """

        self.input_batch_class = namedtuple("InputBatch",
                                            [field.name for field in
                                             dataset.fields if
                                             not field.is_target])

        self.target_batch_class = namedtuple("TargetBatch",
                                             [field.name for field in
                                              dataset.fields if
                                              field.is_target])

        self.batch_size = batch_size
        self.dataset = dataset

        self.shuffle = shuffle

        self.sort_key = sort_key

        self.epoch = 0
        self.iterations = 0

        if self.shuffle:
            if seed is None and internal_random_state is None:
                raise ValueError("If shuffle==True, either seed or "
                                 "internal_random_state have to be != None.")

            self.shuffler = Random(seed)

            if internal_random_state is not None:
                self.shuffler.setstate(internal_random_state)
        else:
            self.shuffler = None

    def __len__(self):
        """ Returns the number of batches this iterator provides in one epoch.

        Returns
        -------
        int
            Number of batches provided in one epoch.
        """

        return math.ceil(len(self.dataset) / self.batch_size)

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
            batch_examples = data[i:i + self.batch_size]
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
            raise RuntimeError(
                "Iterator with shuffle=False does not have "
                "an internal random state."
            )

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
            raise RuntimeError(
                "Iterator with shuffle=False does not have "
                "an internal random state."
            )

        self.shuffler.setstate(state)

    def _create_batch(self, examples):
        # dicts that will be used to create the InputBatch and TargetBatch
        # objects
        input_batch_dict, target_batch_dict = {}, {}

        for field in self.dataset.fields:
            # the length to which all the rows are padded (or truncated)
            pad_length = Iterator._get_pad_length(field, examples)

            # the last batch can have < batch_size examples
            n_rows = min(self.batch_size, len(examples))

            # empty matrix to be filled with numericalized fields
            matrix = np.zeros(shape=(n_rows, pad_length))

            # non-sequential fields all have length = 1, no padding necessary
            should_pad = True if field.sequential else False

            for i, example in enumerate(examples):
                row = field.numericalize(getattr(example, field.name))

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
            indices = list(range(len(self.dataset)))
            self.shuffler.shuffle(indices)

            # creates a new list
            xs = [self.dataset[i] for i in indices]
        else:
            # copies the list
            xs = self.dataset.examples[:]

        # sorting the newly created list
        if self.sort_key is not None:
            xs.sort(key=self.sort_key)

        return xs


class BucketIterator(Iterator):
    """ Creates a bucket iterator that uses a look-ahead heuristic to try and
    batch examples in a way that minimizes the amount of necessary padding.

    It creates a bucket of size N x batch_size, and sorts that bucket before
    splitting it into batches, so there is less padding necessary.
    """

    def __init__(self, dataset, batch_size, sort_key=None, shuffle=True,
                 seed=42, look_ahead_multiplier=100, bucket_sort_key=None):
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
            raise ValueError(
                "For BucketIterator to work, either sort_key or "
                "bucket_sort_key must be != None.")

        super().__init__(dataset, batch_size, sort_key=sort_key,
                         shuffle=shuffle, seed=seed)

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
                bucket = sorted(data[i:i + step], key=self.bucket_sort_key)
            else:
                # if bucket_sort_key is None, sort_key != None so the whole
                # dataset was already sorted with sort_key
                bucket = data[i:i + step]

            for j in range(0, len(bucket), self.batch_size):
                examples = bucket[j:j + self.batch_size]
                input_batch, target_batch = self._create_batch(examples)

                yield input_batch, target_batch
                self.iterations += 1

        # prepare for new epoch
        self.iterations = 0
        self.epoch += 1