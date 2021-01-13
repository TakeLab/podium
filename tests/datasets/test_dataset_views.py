import numpy as np
import pytest

from podium.datasets import (
    Dataset,
    ArrowDataset,
    DatasetBase,
    DatasetConcatView,
    DatasetIndexedView,
    DatasetSlicedView,
    ExampleFactory,
)
from podium.field import Field
from podium.vocab import Vocab


TEST_DATA = [
    (0, "zero"),
    (1, "one"),
    (2, "two"),
    (3, "three"),
    (4, "four"),
    (5, "five"),
    (6, "six"),
    (7, "seven"),
    (8, "eight"),
    (9, "nine"),
]


@pytest.fixture(scope="module")
def fields():
    num_field = Field("number", tokenizer=None)
    name_field = Field("name", numericalizer=Vocab(), is_target=True)
    name_chars_field = Field(
        "name_chars", tokenizer=list, numericalizer=Vocab(), is_target=True
    )
    return [num_field, (name_field, name_chars_field)]


@pytest.fixture(scope="module")
def dataset(fields) -> DatasetBase:
    example_factory = ExampleFactory(fields)
    examples = [example_factory.from_list(e) for e in TEST_DATA]
    ds = Dataset(examples, fields)
    ds.finalize_fields()
    return ds


@pytest.fixture(scope="module")
def dataset_with_upper_field(fields):
    upper_name_field = Field(
        "upper_name", pretokenize_hooks=(str.upper,), numericalizer=Vocab()
    )
    fields = [fields[0], upper_name_field]
    example_factory = ExampleFactory(fields)
    examples = [example_factory.from_list(e) for e in TEST_DATA]
    ds = Dataset(examples, fields)
    ds.finalize_fields()
    return ds


def test_concat_view(dataset):
    dataset_1 = dataset[:5]
    dataset_2 = dataset[5:]

    concat_dataset_1 = DatasetConcatView([dataset_1, dataset_2])

    assert concat_dataset_1.field_dict == dataset_1.field_dict
    assert len(concat_dataset_1) == len(dataset_1) + len(dataset_2)
    assert list(concat_dataset_1.number) == list(dataset.number)

    concat_dataset_2 = DatasetConcatView([dataset_1, dataset_2, dataset_1])

    assert concat_dataset_1.field_dict == dataset_1.field_dict
    assert len(concat_dataset_2) == 2 * len(dataset_1) + len(dataset_2)
    expected = list(dataset_1.number) + list(dataset_2.number) + list(dataset_1.number)
    assert list(concat_dataset_2.number) == expected

    assert concat_dataset_2[11]["number"] == (None, 1)


def test_concat_view_different_fields(dataset, dataset_with_upper_field):
    other_dataset = dataset_with_upper_field

    dataset_concat = DatasetConcatView([dataset, other_dataset])

    assert len(dataset_concat) == len(dataset) + len(other_dataset)
    assert list(dataset_concat.fields) == [dataset.field_dict["number"]]


def test_concat_view_fail_no_field_intersection(dataset):
    upper_name_field = Field(
        "upper_name", pretokenize_hooks=(str.upper,), numericalizer=Vocab()
    )
    fields = [None, upper_name_field]
    example_factory = ExampleFactory(fields)
    examples = [example_factory.from_list(e) for e in TEST_DATA]
    other_dataset = Dataset(examples, fields)
    other_dataset.finalize_fields()

    with pytest.raises(ValueError):
        DatasetConcatView([dataset, other_dataset])


def test_concat_view_override_fields_eager(dataset, fields):
    upper_name_field = Field(
        "name", pretokenize_hooks=(str.upper,), numericalizer=Vocab()
    )
    other_fields = [fields[0], upper_name_field]
    example_factory = ExampleFactory(other_fields)
    examples = [example_factory.from_list(e) for e in TEST_DATA]
    other_dataset = Dataset(examples, other_fields)
    other_dataset.finalize_fields()

    new_field = Field("override_name_field", numericalizer=Vocab(eager=True))
    dataset_concat = DatasetConcatView(
        [dataset, other_dataset], field_overrides={"name": new_field}
    )

    assert dataset_concat.field_dict["override_name_field"].finalized

    concat_vocab = dataset_concat.field_dict["override_name_field"].vocab
    dataset_vocab = dataset.field_dict["name"].vocab
    other_vocab = other_dataset.field_dict["name"].vocab
    assert set(concat_vocab.itos) == set(dataset_vocab.itos) | set(other_vocab.itos)


def test_concat_view_override_fields_non_eager(dataset, fields):
    upper_name_field = Field(
        "name", pretokenize_hooks=(str.upper,), numericalizer=Vocab()
    )
    other_fields = [fields[0], upper_name_field]
    example_factory = ExampleFactory(other_fields)
    examples = [example_factory.from_list(e) for e in TEST_DATA]
    other_dataset = Dataset(examples, other_fields)
    other_dataset.finalize_fields()

    new_field = Field("override_name_field", numericalizer=Vocab(eager=False))
    dataset_concat = DatasetConcatView(
        [dataset, other_dataset], field_overrides={"name": new_field}
    )

    assert not dataset_concat.field_dict["override_name_field"].finalized

    dataset_concat.finalize_fields()
    concat_vocab = dataset_concat.field_dict["override_name_field"].vocab
    dataset_vocab = dataset.field_dict["name"].vocab
    other_vocab = other_dataset.field_dict["name"].vocab
    assert set(concat_vocab.itos) == set(dataset_vocab.itos) | set(other_vocab.itos)


def test_concat_view_batching(dataset):
    dataset_1 = dataset[:5]
    dataset_2 = dataset[5:]

    concat_dataset = DatasetConcatView([dataset_1, dataset_2])

    input_batch, target_batch = concat_dataset.batch()

    expected = [[x] for x in range(10)]
    assert np.all(input_batch.number == expected)


def test_indexed_view(dataset):
    indices = [1, 5, 6, 5]
    dataset_view = DatasetIndexedView(dataset, indices=indices)

    # int indexing
    for view_index, real_index in enumerate(indices):
        assert dataset_view[view_index] == dataset[real_index]

    # iteration
    for real_index, view_example in zip(indices, dataset_view):
        assert view_example == dataset[real_index]


def test_indexed_view_batching(dataset):
    indices = [1, 5, 6, 5]
    dataset_view = DatasetIndexedView(dataset, indices=indices)

    view_input_batch, view_target_batch = dataset_view.batch()
    dataset_input_batch, dataset_target_batch = dataset.batch()

    assert len(view_input_batch) == 1
    assert len(view_target_batch) == 2

    assert np.all(view_input_batch.number == dataset_input_batch.number[indices])
    assert np.all(view_target_batch.name == dataset_target_batch.name[indices])


def test_sliced_view(dataset):
    start, stop, step = 3, 8, 2
    indices = list(range(start, stop, step))
    slc = slice(start, stop, step)
    dataset_view = DatasetSlicedView(dataset, s=slc)

    # int indexing
    for view_index, real_index in enumerate(indices):
        assert dataset_view[view_index] == dataset[real_index]

    # interation
    for view_ex, real_index in zip(dataset_view, range(start, stop, step)):
        assert view_ex == dataset[real_index]

    # test negative step
    start, stop, step = 8, 3, -2
    indices = list(range(start, stop, step))
    slc = slice(start, stop, step)
    dataset_view = DatasetSlicedView(dataset, s=slc)

    # int indexing
    for view_index, real_index in enumerate(indices):
        assert dataset_view[view_index] == dataset[real_index]

    # interation
    for view_ex, real_index in zip(dataset_view, range(start, stop, step)):
        assert view_ex == dataset[real_index]


def test_sliced_view_batching(dataset):
    start, stop, step = 3, 8, 2
    slc = slice(start, stop, step)
    indices = list(range(start, stop, step))
    dataset_view = DatasetSlicedView(dataset, s=slc)

    view_input_batch, view_target_batch = dataset_view.batch()
    dataset_input_batch, dataset_target_batch = dataset.batch()

    assert len(view_input_batch) == 1
    assert len(view_target_batch) == 2

    assert np.all(view_input_batch.number == dataset_input_batch.number[indices])
    assert np.all(view_target_batch.name == dataset_target_batch.name[indices])

    # test negative step
    start, stop, step = 8, 3, -2
    slc = slice(start, stop, step)
    indices = list(range(start, stop, step))
    dataset_view = DatasetSlicedView(dataset, s=slc)

    view_input_batch, view_target_batch = dataset_view.batch()

    assert len(view_input_batch) == 1
    assert len(view_target_batch) == 2

    assert np.all(view_input_batch.number == dataset_input_batch.number[indices])
    assert np.all(view_target_batch.name == dataset_target_batch.name[indices])


def test_slice_view_to_dataset(dataset, tmp_path):
    start, stop, step = 3, 8, 2
    slc = slice(start, stop, step)
    dataset_view = DatasetSlicedView(dataset, s=slc)

    # cast to Dataset
    ds = Dataset.from_dataset(dataset_view)
    assert isinstance(ds, Dataset)
    assert len(ds) == len(dataset_view)
    for ex_view, ex_dataset in zip(dataset_view, ds):
        assert ex_view == ex_dataset

    # cast to ArrowDataset
    ds = ArrowDataset.from_dataset(dataset_view, cache_path=tmp_path)
    assert isinstance(ds, ArrowDataset)
    assert len(ds) == len(dataset_view)
    for ex_view, ex_dataset in zip(dataset_view, ds):
        assert ex_view == ex_dataset
