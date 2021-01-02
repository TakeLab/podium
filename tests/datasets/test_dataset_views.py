import pytest

from podium.datasets import Dataset, DatasetABC, DatasetConcatView
from podium.storage import ExampleFactory, Field, Vocab


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
    name_field = Field("name", numericalizer=Vocab())
    name_chars_field = Field("name_chars", tokenizer=list, numericalizer=Vocab())
    return [num_field, (name_field, name_chars_field)]


@pytest.fixture(scope="module")
def dataset(fields) -> DatasetABC:
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


def test_concat_view_different_fields(dataset, dataset_with_upper_field):
    other_dataset = dataset_with_upper_field

    dataset_concat = DatasetConcatView([dataset, other_dataset])

    assert len(dataset_concat) == len(dataset) + len(other_dataset)
    assert list(dataset_concat.fields) == [dataset.field_dict["number"]]


def test_concat_view_fail_no_field_intersection(dataset, dataset_with_upper_field):
    other_dataset = dataset_with_upper_field

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
