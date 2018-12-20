import os
import random

import pytest
import numpy as np

from takepod.storage.dataset import TabularDataset
from takepod.storage.field import Field
from takepod.storage.iterator import Iterator, BucketIterator
from takepod.storage.vocab import Vocab
from .conftest import create_temp_json


@pytest.mark.parametrize(
    "batch_size, expected_len",
    [
        (1, 7),
        (2, 4),
        (3, 3),
        (4, 2),
        (6, 2),
        (7, 1),
        (10, 1),
    ]
)
def test_len(batch_size, expected_len, dataset):
    dataset.finalize_fields()

    iterator = Iterator(dataset, batch_size)

    assert expected_len == len(iterator)


@pytest.mark.parametrize(
    "fixed_length, expected_shape",
    [
        (None, (7, 6)),
        (1, (7, 1)),
        (3, (7, 3)),
        (5, (7, 5)),
        (7, (7, 7))
    ]
)
def test_padding(fixed_length, expected_shape, file_path):
    fields = tabular_dataset_fields(fixed_length)
    ds = create_tabular_dataset_from_json(fields, file_path)

    batch_size = 7
    ds.finalize_fields()

    iterator = Iterator(ds, batch_size=batch_size, shuffle=False)

    input_batch, _ = next(iter(iterator))

    assert input_batch.text.shape == expected_shape

    PAD_SYMBOL = fields["text"].vocab.pad_symbol()

    for i, row in enumerate(input_batch.text):
        n_el = len(TABULAR_TEXT[i].split())

        assert (row[:n_el].astype(np.int32) != PAD_SYMBOL).all()
        assert (row[n_el:].astype(np.int32) == PAD_SYMBOL).all()


def test_iterate_new_epoch(dataset):
    dataset.finalize_fields()

    iterator = Iterator(dataset, 2)

    it = iter(iterator)
    assert iterator.iterations == 0

    for i in range(4):
        next(it)
        assert iterator.epoch == 0
        assert iterator.iterations == i

    with pytest.raises(StopIteration):
        next(it)

    assert iterator.epoch == 1
    assert iterator.iterations == 0


def test_create_batch(dataset):
    expected_row_lengths = [3, 4, 3, 6]

    dataset.finalize_fields()
    batch_size = 2
    iterator = Iterator(dataset, batch_size=batch_size, shuffle=False)

    iter_len = len(iterator)
    assert iter_len == 4
    for i, ((x_batch, y_batch), expected_row_length) in enumerate(
            zip(iterator, expected_row_lengths)):
        assert hasattr(x_batch, "text") and not hasattr(x_batch, "rating")
        assert hasattr(y_batch, "rating") and not hasattr(y_batch, "text")

        assert x_batch.text.shape[1] == expected_row_length
        assert y_batch.rating.shape[1] == 1

        if (i + 1) == iter_len:
            assert x_batch.text.shape[0] == 1
            assert y_batch.rating.shape[0] == 1
        else:
            assert x_batch.text.shape[0] == batch_size
            assert y_batch.rating.shape[0] == batch_size


def test_sort_key(dataset):
    dataset.finalize_fields()

    def text_len_sort_key(example):
        tokens = example.text[1]
        return len(tokens)

    iterator = Iterator(dataset, batch_size=2, sort_key=text_len_sort_key,
                        shuffle=False)

    expected_row_lengths = [1, 3, 4, 6]

    for (x_batch, y_batch), expected_row_length in zip(iterator,
                                                       expected_row_lengths):
        assert x_batch.text.shape[1] == expected_row_length


@pytest.mark.parametrize(
    "seed_1, seed_2, num_epochs_1, num_epochs_2, expect_identical_behaviour",
    [
        (1, 2, 3, 3, False),
        (1, 2, 3, 4, False),
        (1, 1, 3, 3, True),  # expect identical behaviour only in the case of
                             # same seeds and same number of epochs elapsed
        (1, 1, 3, 4, False),
    ]
)
def test_shuffle_deterministic_sequence(seed_1, seed_2, num_epochs_1,
                                        num_epochs_2,
                                        expect_identical_behaviour, dataset):
    dataset.finalize_fields()

    random.seed(42)  # internal random state independent from global seed

    iterator = Iterator(dataset, batch_size=2, shuffle=True, seed=seed_1)
    run_n_epochs(iterator, num_epochs_1)  # iterate for num_epochs_1 epochs

    random.seed(43)  # internal random state independent from global seed

    iterator_2 = Iterator(dataset, batch_size=2, shuffle=True, seed=seed_2)
    run_n_epochs(iterator_2, num_epochs_2)  # iterate for num_epochs_2 epochs

    random.seed(44)  # internal random state independent from global seed

    if expect_identical_behaviour:
        assert iterators_behave_identically(iterator, iterator_2)
    else:
        # Beware, for some combination of different seeds and numbers of
        # epochs the iterators might actually behave identically.
        # For the chosen combination they don't.
        assert not iterators_behave_identically(iterator, iterator_2)


def test_shuffle_random_state(dataset):
    dataset.finalize_fields()

    random.seed(5)  # internal random state independent from global seed
    # run first iterator for 3 epochs
    iterator = Iterator(dataset, batch_size=2, shuffle=True)
    run_n_epochs(iterator, 3)

    # get first iterator's internal state
    state = iterator.get_internal_random_state()

    random.seed(6)  # internal random state independent from global seed

    # initialize second iterator with the state
    iterator_2 = Iterator(dataset, batch_size=2, shuffle=True,
                          internal_random_state=state)

    # run both iterators for 2 epochs
    run_n_epochs(iterator, 2)
    random.seed(8)  # internal random state independent from global seed
    run_n_epochs(iterator_2, 2)

    # the iterators should behave identically
    assert iterators_behave_identically(iterator, iterator_2)

    iterator_3 = Iterator(dataset, batch_size=2, shuffle=True)
    iterator_3.set_internal_random_state(
        iterator_2.get_internal_random_state())

    # the iterators should behave identically
    assert iterators_behave_identically(iterator_2, iterator_3)


def test_shuffle_no_seed_or_state_exception(dataset):
    dataset.finalize_fields()

    with pytest.raises(ValueError):
        Iterator(dataset, batch_size=2, shuffle=True, seed=None,
                 internal_random_state=None)


def test_shuffle_random_state_exception(dataset):
    dataset.finalize_fields()

    iterator = Iterator(dataset, batch_size=2, shuffle=False)

    with pytest.raises(RuntimeError):
        iterator.get_internal_random_state()

    iterator_2 = Iterator(dataset, batch_size=2, shuffle=True)
    state = iterator_2.get_internal_random_state()

    with pytest.raises(RuntimeError):
        iterator.set_internal_random_state(state)


@pytest.mark.parametrize(
    "train, given_shuffle, expected_shuffle",
    [
        (False, None, False),  # if no shuffle given, fall back to train
        (True, None, True),
        (False, False, False),  # if shuffle given, set to that value
        (True, False, False),
        (False, True, True),
        (True, True, True),
    ]
)
def test_shuffle_correct_values(train, given_shuffle, expected_shuffle,
                                dataset):
    dataset.finalize_fields()
    iterator = Iterator(dataset, batch_size=2, shuffle=given_shuffle,
                        train=train)

    assert iterator.shuffle == expected_shuffle


def text_len_key(example):
    return len(example.text[1])


@pytest.mark.parametrize(
    "look_ahead_multiplier, expected_row_lengths, bucket_sort_key, sort_key",
    [
        # effectively, no look-ahead
        (1, [3, 4, 3, 6], text_len_key, None),
        (2, [1, 4, 3, 6], text_len_key, None),

        # effectively, sort the whole dataset
        (4, [1, 3, 4, 6], text_len_key, None),
        (8, [1, 3, 4, 6], text_len_key, None),

        # if sort_key is set, the whole dataset is sorted
        (1, [1, 3, 4, 6], None, text_len_key),
        (2, [1, 3, 4, 6], None, text_len_key),
        (4, [1, 3, 4, 6], None, text_len_key),
    ]
)
def test_bucket_iterator(look_ahead_multiplier, expected_row_lengths,
                         bucket_sort_key, sort_key, dataset):
    dataset.finalize_fields()

    iterator = BucketIterator(
        dataset,
        batch_size=2,
        shuffle=False,
        sort_key=sort_key,
        bucket_sort_key=bucket_sort_key,
        look_ahead_multiplier=look_ahead_multiplier
    )

    for (x_batch, y_batch), expected_row_length in zip(iterator,
                                                       expected_row_lengths):
        assert x_batch.text.shape[1] == expected_row_length


def test_bucket_iterator_exception(dataset):
    dataset.finalize_fields()

    with pytest.raises(ValueError):
        BucketIterator(dataset, batch_size=2, sort_key=None,
                       bucket_sort_key=None, look_ahead_multiplier=2)


def iterators_behave_identically(iterator_1, iterator_2):
    all_equal = True

    for (x_batch_1, y_batch_1), (x_batch_2, y_batch_2) in zip(iterator_1,
                                                              iterator_2):

        x_equal = np_arrays_equal(x_batch_1.text, x_batch_2.text)
        y_equal = np_arrays_equal(y_batch_1.rating, y_batch_2.rating)

        equal = x_equal and y_equal

        if not equal:
            # if any batch is different, the iterators
            # don't have identical behaviour
            all_equal = False
            break

    return all_equal


def run_n_epochs(iterator, num_epochs):
    for _ in range(num_epochs):
        for _ in iterator:
            pass


def np_arrays_equal(arr_1, arr_2):
    arrs_equal = (arr_1 == arr_2)

    if isinstance(arrs_equal, np.ndarray):
        arrs_equal = arrs_equal.all()

    return arrs_equal


@pytest.fixture()
def vocab(tabular_dataset_fields):
    return tabular_dataset_fields["text"].vocab


@pytest.fixture()
def dataset(file_path, tabular_dataset_fields):
    return create_tabular_dataset_from_json(tabular_dataset_fields, file_path)


@pytest.fixture()
def tabular_dataset_fields(fixed_length=None):
    TEXT = Field('text', eager=True, vocab=Vocab(), fixed_length=fixed_length)
    RATING = Field('rating', sequential=False, eager=False, is_target=True)

    fields = {"text": TEXT, "rating": RATING}

    return fields


TABULAR_TEXT = (
    "a b c",
    "a",
    "a b c d",
    "a",
    "d b",
    "d c g",
    "b b b b b b",
)

TABULAR_RATINGS = (2.5, 3.2, 1.1, 2.1, 5.4, 2.8, 1.9)


@pytest.fixture()
def tabular_data():
    return {
        "text": TABULAR_TEXT,
        "rating": TABULAR_RATINGS,
    }


@pytest.fixture()
def file_path(tmpdir, tabular_data):
    # tmpdir is a default pytest fixture
    path = os.path.join(tmpdir, "sample.json")

    create_temp_json(path, tabular_data)

    yield path


def create_tabular_dataset_from_json(fields, file_path):
    return TabularDataset(file_path, "json", fields, skip_header=False)
