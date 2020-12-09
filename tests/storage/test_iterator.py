import random

import numpy as np
import pytest

from podium.datasets import Dataset
from podium.datasets.hierarhical_dataset import HierarchicalDataset
from podium.datasets.iterator import (
    BucketIterator,
    HierarchicalDatasetIterator,
    Iterator,
    SingleBatchIterator,
)
from podium.storage import ExampleFactory, Field, Vocab

from .conftest import (
    TABULAR_TEXT,
    create_tabular_dataset_from_json,
    tabular_dataset_fields,
)


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
    ],
)
@pytest.mark.usefixtures("tabular_dataset")
def test_len(batch_size, expected_len, tabular_dataset):
    tabular_dataset.finalize_fields()

    iterator = Iterator(dataset=tabular_dataset, batch_size=batch_size)

    assert len(iterator) == expected_len


@pytest.mark.parametrize(
    "fixed_length, expected_shape",
    [(None, (7, 6)), (1, (7, 1)), (3, (7, 3)), (5, (7, 5)), (7, (7, 7))],
)
@pytest.mark.usefixtures("json_file_path")
def test_padding(fixed_length, expected_shape, json_file_path):
    fields = tabular_dataset_fields(fixed_length=fixed_length)
    ds = create_tabular_dataset_from_json(fields=fields, json_file_path=json_file_path)

    batch_size = 7
    ds.finalize_fields()

    iterator = Iterator(dataset=ds, batch_size=batch_size, shuffle=False)

    input_batch, _ = next(iter(iterator))

    assert input_batch.text.shape == expected_shape

    pad_symbol = fields["text"].vocab.padding_index()

    for i, row in enumerate(input_batch.text):
        if TABULAR_TEXT[i] is None:
            # if missing data
            continue

        n_el = len(TABULAR_TEXT[i].split())

        assert (row[:n_el].astype(np.int32) != pad_symbol).all()
        assert (row[n_el:].astype(np.int32) == pad_symbol).all()


@pytest.mark.usefixtures("tabular_dataset")
def test_iterate_new_epoch(tabular_dataset):
    tabular_dataset.finalize_fields()

    iterator = Iterator(dataset=tabular_dataset, batch_size=2)

    it = iter(iterator)
    assert iterator._iterations == 0

    for i in range(4):
        next(it)
        assert iterator._epoch == 0
        assert iterator._iterations == i

    with pytest.raises(StopIteration):
        next(it)

    assert iterator._epoch == 1
    assert iterator._iterations == 0


@pytest.mark.usefixtures("tabular_dataset")
def test_create_batch(tabular_dataset):
    expected_row_lengths = [3, 4, 3, 6]

    tabular_dataset.finalize_fields()
    batch_size = 2
    iterator = Iterator(dataset=tabular_dataset, batch_size=batch_size, shuffle=False)

    iter_len = len(iterator)
    assert iter_len == 4
    for i, ((x_batch, y_batch), expected_row_length) in enumerate(
        zip(iterator, expected_row_lengths)
    ):
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


@pytest.mark.usefixtures("json_file_path")
def test_not_numericalizable_field(json_file_path):
    class MockCustomDataClass:
        def __init__(self, data):
            self.data = data

    def custom_datatype_tokenizer(data):
        return MockCustomDataClass(data)

    fields = tabular_dataset_fields()
    text_field = fields["text_with_missing_data"]
    non_numericalizable_field = Field(
        "non_numericalizable_field",
        tokenizer=custom_datatype_tokenizer,
        numericalizer=None,
        allow_missing_data=True,
        keep_raw=True,
    )

    fields["text_with_missing_data"] = (text_field, non_numericalizable_field)

    dataset = create_tabular_dataset_from_json(fields, json_file_path)
    dataset.finalize_fields()

    for x_batch, _ in Iterator(dataset, batch_size=len(dataset), shuffle=False):
        assert isinstance(x_batch.non_numericalizable_field, (list, tuple))
        for i, batch_data, real_data in zip(
            range(len(dataset)), x_batch.non_numericalizable_field, TABULAR_TEXT
        ):
            if i == 3:
                assert batch_data is None
            else:
                assert isinstance(batch_data, MockCustomDataClass)
                assert batch_data.data == real_data


@pytest.mark.usefixtures("tabular_dataset")
def test_lazy_numericalization_caching(tabular_dataset):
    tabular_dataset.finalize_fields()

    # Run one epoch to cause lazy numericalization
    for _ in Iterator(dataset=tabular_dataset, batch_size=10):
        pass

    # Test if cached data is equal to numericalized data
    for example in tabular_dataset:
        for field in tabular_dataset.fields:
            example_data = example[field.name]
            numericalized_data = field.numericalize(example_data)

            cached_data = example[f"{field.name}_"]
            assert np.all(numericalized_data == cached_data)


@pytest.mark.usefixtures("tabular_dataset")
def test_sort_key(tabular_dataset):
    tabular_dataset.finalize_fields()

    def text_len_sort_key(example):
        tokens = example["text"][1]
        if tokens is None:
            return 0
        else:
            return len(tokens)

    iterator = Iterator(
        dataset=tabular_dataset, batch_size=2, sort_key=text_len_sort_key, shuffle=False
    )

    expected_row_lengths = [1, 3, 4, 6]

    for (x_batch, _), expected_row_length in zip(iterator, expected_row_lengths):
        assert x_batch.text.shape[1] == expected_row_length


@pytest.mark.parametrize(
    "seed_1, seed_2, num_epochs_1, num_epochs_2, expect_identical_behaviour",
    [
        (1, 2, 3, 3, False),
        (1, 2, 3, 4, False),
        (1, 1, 3, 3, True),  # expect identical behaviour only in the case of
        # same seeds and same number of epochs elapsed
        (1, 1, 3, 4, False),
    ],
)
@pytest.mark.usefixtures("tabular_dataset")
def test_shuffle_deterministic_sequence(
    seed_1,
    seed_2,
    num_epochs_1,
    num_epochs_2,
    expect_identical_behaviour,
    tabular_dataset,
):
    tabular_dataset.finalize_fields()

    random.seed(42)  # internal random state independent from global seed

    iterator = Iterator(dataset=tabular_dataset, batch_size=2, shuffle=True, seed=seed_1)
    run_n_epochs(iterator, num_epochs_1)  # iterate for num_epochs_1 epochs

    random.seed(43)  # internal random state independent from global seed

    iterator_2 = Iterator(
        dataset=tabular_dataset, batch_size=2, shuffle=True, seed=seed_2
    )
    run_n_epochs(iterator_2, num_epochs_2)  # iterate for num_epochs_2 epochs

    random.seed(44)  # internal random state independent from global seed

    if expect_identical_behaviour:
        assert iterators_behave_identically(iterator, iterator_2)
    else:
        # Beware, for some combination of different seeds and numbers of
        # epochs the iterators might actually behave identically.
        # For the chosen combination they don't.
        assert not iterators_behave_identically(iterator, iterator_2)


@pytest.mark.usefixtures("tabular_dataset")
def test_shuffle_random_state(tabular_dataset):
    tabular_dataset.finalize_fields()

    random.seed(5)  # internal random state independent from global seed
    # run first iterator for 3 epochs
    iterator = Iterator(dataset=tabular_dataset, batch_size=2, shuffle=True)
    run_n_epochs(iterator, 3)

    # get first iterator's internal state
    state = iterator.get_internal_random_state()

    random.seed(6)  # internal random state independent from global seed

    # initialize second iterator with the state
    iterator_2 = Iterator(
        dataset=tabular_dataset, batch_size=2, shuffle=True, internal_random_state=state
    )

    # run both iterators for 2 epochs
    run_n_epochs(iterator, 2)
    random.seed(8)  # internal random state independent from global seed
    run_n_epochs(iterator_2, 2)

    # the iterators should behave identically
    assert iterators_behave_identically(iterator, iterator_2)

    iterator_3 = Iterator(dataset=tabular_dataset, batch_size=2, shuffle=True)
    iterator_3.set_internal_random_state(iterator_2.get_internal_random_state())

    # the iterators should behave identically
    assert iterators_behave_identically(iterator_2, iterator_3)


@pytest.mark.usefixtures("tabular_dataset")
def test_shuffle_no_seed_or_state_exception(tabular_dataset):
    tabular_dataset.finalize_fields()

    with pytest.raises(ValueError):
        Iterator(
            dataset=tabular_dataset,
            batch_size=2,
            shuffle=True,
            seed=None,
            internal_random_state=None,
        )


@pytest.mark.usefixtures("tabular_dataset")
def test_shuffle_random_state_exception(tabular_dataset):
    tabular_dataset.finalize_fields()

    iterator = Iterator(dataset=tabular_dataset, batch_size=2, shuffle=False)

    with pytest.raises(RuntimeError):
        iterator.get_internal_random_state()

    iterator_2 = Iterator(dataset=tabular_dataset, batch_size=2, shuffle=True)
    state = iterator_2.get_internal_random_state()

    with pytest.raises(RuntimeError):
        iterator.set_internal_random_state(state)


def text_len_key(example):
    if example["text"][1] is None:
        return 0
    else:
        return len(example["text"][1])


@pytest.mark.usefixtures("json_file_path")
def test_iterator_missing_data_in_batch(json_file_path):
    missing_data_default_value = -99
    fields = tabular_dataset_fields()
    missing_value_field = Field(
        "missing_value_field",
        tokenizer="split",
        numericalizer=Vocab(),
        allow_missing_data=True,
        keep_raw=True,
        missing_data_token=missing_data_default_value,
    )
    fields["text_with_missing_data"] = missing_value_field
    ds = create_tabular_dataset_from_json(fields, json_file_path)

    for x_batch, _ in Iterator(ds, batch_size=len(ds), shuffle=False):
        # test if the value we know is missing is correctly filled out
        missing_value_row = x_batch.missing_value_field[3]
        assert np.all(missing_value_row == missing_data_default_value)


@pytest.mark.usefixtures("tabular_dataset")
def test_single_batch_iterator(tabular_dataset):
    single_batch_iterator = SingleBatchIterator(tabular_dataset)

    assert single_batch_iterator._batch_size == len(tabular_dataset)
    assert len(single_batch_iterator) == 1
    for i, (input_batch, target_batch) in enumerate(single_batch_iterator):
        assert i == 0, "Multiple batches from SingleBatchIterator"
        assert len(input_batch.text) == 7
        assert len(target_batch.rating) == 7

    sliced_dataset = tabular_dataset[:5]
    for i, (input_batch, target_batch) in enumerate(
        single_batch_iterator(sliced_dataset)
    ):
        assert i == 0, "Multiple batches from SingleBatchIterator"
        assert single_batch_iterator._batch_size == len(sliced_dataset)
        assert len(input_batch.text) == 5
        assert len(target_batch.rating) == 5


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
    ],
)
@pytest.mark.usefixtures("tabular_dataset")
def test_bucket_iterator(
    look_ahead_multiplier,
    expected_row_lengths,
    bucket_sort_key,
    sort_key,
    tabular_dataset,
):
    tabular_dataset.finalize_fields()

    iterator = BucketIterator(
        dataset=tabular_dataset,
        batch_size=2,
        shuffle=False,
        sort_key=sort_key,
        bucket_sort_key=bucket_sort_key,
        look_ahead_multiplier=look_ahead_multiplier,
    )

    for (x_batch, _), expected_row_length in zip(iterator, expected_row_lengths):
        assert x_batch.text.shape[1] == expected_row_length


@pytest.mark.usefixtures("tabular_dataset")
def test_bucket_iterator_exception(tabular_dataset):
    tabular_dataset.finalize_fields()

    with pytest.raises(ValueError):
        BucketIterator(
            dataset=tabular_dataset,
            batch_size=2,
            sort_key=None,
            bucket_sort_key=None,
            look_ahead_multiplier=2,
        )


@pytest.mark.usefixtures("tabular_dataset")
def test_bucket_iterator_no_dataset_on_init(tabular_dataset):
    tabular_dataset.finalize_fields()

    bi = BucketIterator(
        dataset=None,
        batch_size=2,
        sort_key=None,
        bucket_sort_key=text_len_key,
        look_ahead_multiplier=2,
    )
    # since no dataset is set, one can not iterate
    with pytest.raises(TypeError):
        for x_batch, y_batch in bi:
            pass


@pytest.mark.usefixtures("tabular_dataset")
def test_bucket_iterator_set_dataset_on_init(tabular_dataset):
    tabular_dataset.finalize_fields()

    bi = BucketIterator(
        dataset=None,
        batch_size=2,
        sort_key=None,
        bucket_sort_key=text_len_key,
        look_ahead_multiplier=2,
    )
    # setting dataset
    bi.set_dataset(tabular_dataset)
    # iterating over dataset
    for x_batch, y_batch in bi:
        # asserting to iterate
        assert True


def test_iterator_batch_as_list():
    raw_dataset = [("1 2 3 4",), ("2 3 4",), ("3 4",)]
    field = Field(
        "test_field", numericalizer=int, tokenizer="split", disable_batch_matrix=True
    )
    fields = (field,)
    ef = ExampleFactory(fields)
    examples = [ef.from_list(raw_example) for raw_example in raw_dataset]
    ds = Dataset(examples, fields)

    for i, (input_batch, _) in enumerate(Iterator(ds, batch_size=2, shuffle=False)):
        assert isinstance(input_batch.test_field, list)
        batch = input_batch.test_field
        if i == 0:
            assert len(batch) == 2
            assert np.all(batch[0] == [1, 2, 3, 4])
            assert np.all(batch[1] == [2, 3, 4])

        if i == 2:
            assert len(batch) == 1
            assert np.all(batch[0] == [3, 4])


def iterators_behave_identically(iterator_1, iterator_2):
    all_equal = True

    for (x_batch_1, y_batch_1), (x_batch_2, y_batch_2) in zip(iterator_1, iterator_2):

        x_equal = np.array_equal(x_batch_1.text, x_batch_2.text)
        y_equal = np.array_equal(y_batch_1.rating, y_batch_2.rating)

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


@pytest.fixture
def hierarchical_dataset_fields():
    name_field = Field(name="name", keep_raw=True, tokenizer=None, numericalizer=Vocab())
    number_field = Field(name="number", keep_raw=True, tokenizer=None, numericalizer=int)

    fields = {"name": name_field, "number": number_field}
    return fields


@pytest.fixture
def hierarchical_dataset_parser():
    return HierarchicalDataset.get_default_dict_parser("children")


@pytest.fixture
def hierarchical_dataset(hierarchical_dataset_fields, hierarchical_dataset_parser):
    dataset = HierarchicalDataset.from_json(
        HIERARCHIAL_DATASET_JSON_EXAMPLE,
        hierarchical_dataset_fields,
        hierarchical_dataset_parser,
    )
    dataset.finalize_fields()
    return dataset

@pytest.fixture
def hierarchical_dataset_2(hierarchical_dataset_fields, hierarchical_dataset_parser):
    dataset = HierarchicalDataset.from_json(
        HIERARCHIAL_DATASET_JSON_EXAMPLE_2,
        hierarchical_dataset_fields,
        hierarchical_dataset_parser,
    )
    dataset.finalize_fields()
    return dataset

def test_hierarchical_dataset_iteration(hierarchical_dataset):
    hit = HierarchicalDatasetIterator(dataset=hierarchical_dataset, batch_size=3)
    batch_iter = iter(hit)

    input_batch_1, _ = next(batch_iter)
    assert len(input_batch_1.number) == 3
    assert np.all(input_batch_1.number[0] == [[1]])
    assert np.all(input_batch_1.number[1] == [[1], [2]])
    assert np.all(input_batch_1.number[2] == [[1], [2], [3]])

    input_batch_2, _ = next(batch_iter)
    assert len(input_batch_2.number) == 3
    assert np.all(input_batch_2.number[0] == [[1], [2], [4]])
    assert np.all(input_batch_2.number[1] == [[5]])
    assert np.all(input_batch_2.number[2] == [[5], [6]])

    input_batch_3, _ = next(batch_iter)
    assert len(input_batch_1.number) == 3
    assert np.all(input_batch_3.number[0] == [[5], [6], [7]])
    assert np.all(input_batch_3.number[1] == [[5], [6], [7], [8]])
    assert np.all(input_batch_3.number[2] == [[5], [6], [7], [8], [9]])

    input_batch_4, _ = next(batch_iter)
    assert len(input_batch_4.number) == 1
    assert np.all(input_batch_4.number[0] == [[5], [6], [7], [8], [10]])

    with pytest.raises(StopIteration):
        next(batch_iter)


def test_hierarchical_dataset_iteration_with_depth_limitation(hierarchical_dataset):
    hit = HierarchicalDatasetIterator(
        dataset=hierarchical_dataset, batch_size=20, context_max_depth=0
    )
    batch_iter = iter(hit)

    input_batch, _ = next(batch_iter)
    assert np.all(input_batch.number[2] == [[2], [3]])
    assert np.all(input_batch.number[8] == [[8], [9]])


def test_hierarchial_dataset_iterator_numericalization_caching(hierarchical_dataset):
    # Run one epoch to cause lazy numericalization
    hit = HierarchicalDatasetIterator(
        dataset=hierarchical_dataset, batch_size=20, context_max_depth=2
    )
    for _ in hit:
        pass

    # Test if cached data is equal to numericalized data
    for example in hierarchical_dataset:
        for field in hierarchical_dataset.fields:
            example_data = example[field.name]
            numericalized_data = field.numericalize(example_data)

            cached_data = example[f"{field.name}_"]
            assert np.all(numericalized_data == cached_data)


def test_hierarchical_no_dataset_set():
    hi = HierarchicalDatasetIterator(batch_size=20, context_max_depth=2)
    with pytest.raises(AttributeError):
        for b in hi:
            pass


def test_hierarchical_set_dataset_after(hierarchical_dataset, hierarchical_dataset_2):
    hi = HierarchicalDatasetIterator(batch_size=3, context_max_depth=2)
    hi.set_dataset(hierarchical_dataset)
    batch_iter = iter(hi)

    input_batch_1, _ = next(batch_iter)
    assert len(input_batch_1.number) == 3
    assert np.all(input_batch_1.number[0] == [[1]])
    assert np.all(input_batch_1.number[1] == [[1], [2]])
    assert np.all(input_batch_1.number[2] == [[1], [2], [3]])

    input_batch_2, _ = next(batch_iter)
    assert len(input_batch_2.number) == 3
    assert np.all(input_batch_2.number[0] == [[1], [2], [4]])
    assert np.all(input_batch_2.number[1] == [[5]])
    assert np.all(input_batch_2.number[2] == [[5], [6]])

    input_batch_3, _ = next(batch_iter)
    assert len(input_batch_1.number) == 3
    assert np.all(input_batch_3.number[0] == [[5], [6], [7]])
    assert np.all(input_batch_3.number[1] == [[5], [6], [7], [8]])
    assert np.all(input_batch_3.number[2] == [[5], [6], [7], [8], [9]])

    input_batch_4, _ = next(batch_iter)
    assert len(input_batch_4.number) == 1
    assert np.all(input_batch_4.number[0] == [[5], [6], [7], [8], [10]])

    with pytest.raises(StopIteration):
        next(batch_iter)

    # change dataset
    hi.set_dataset(hierarchical_dataset_2)
    batch_iter = iter(hi)

    input_batch_1, _ = next(batch_iter)
    assert len(input_batch_1.number) == 3
    assert np.all(input_batch_1.number[0] == [[1]])
    assert np.all(input_batch_1.number[1] == [[1], [2]])
    assert np.all(input_batch_1.number[2] == [[1], [2], [3]])

    input_batch_2, _ = next(batch_iter)
    assert len(input_batch_2.number) == 3
    assert np.all(input_batch_2.number[0] == [[5]])
    assert np.all(input_batch_2.number[1] == [[5], [6]])
    assert np.all(input_batch_2.number[2] == [[5], [6], [7]])

    input_batch_3, _ = next(batch_iter)
    assert len(input_batch_3.number) == 1
    assert np.all(input_batch_3.number[0] == [[5], [6], [7], [10]])

    with pytest.raises(StopIteration):
        next(batch_iter)

def test_hierarchical_change_dataset(hierarchial_dataset, hierarchical_dataset_2):
    pass

HIERARCHIAL_DATASET_JSON_EXAMPLE = """
[
{
    "name" : "parent1",
    "number" : 1,
    "children" : [
        {
            "name" : "c11",
            "number" : 2,
            "children" : [
                {
                    "name" : "c111",
                    "number" : 3
                }
            ]
        },
        {
            "name" : "c12",
            "number" : 4
        }
    ]
},
{
    "name" : "parent2",
    "number" : 5,
    "children" : [
        {
            "name" : "c21",
            "number" : 6
        },
        {
            "name" : "c22",
            "number" : 7,
            "children" : []
        },
        {
            "name" : "c23",
            "number" : 8,
            "children" : [
                {
                    "name" : "c231",
                    "number" : 9
                }
            ]
        },
        {
            "name" : "c24",
            "number" : 10
        }
    ]
}
]
"""

HIERARCHIAL_DATASET_JSON_EXAMPLE_2 = """
[
{
    "name" : "parent1",
    "number" : 1,
    "children" : [
        {
            "name" : "c11",
            "number" : 2,
            "children" : [
                {
                    "name" : "c111",
                    "number" : 3
                }
            ]
        }
    ]
},
{
    "name" : "parent2",
    "number" : 5,
    "children" : [
        {
            "name" : "c21",
            "number" : 6
        },
        {
            "name" : "c22",
            "number" : 7,
            "children" : []
        },
        {
            "name" : "c24",
            "number" : 10
        }
    ]
}
]
"""