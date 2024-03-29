import csv
import os
import tempfile
from functools import partial

import numpy as np
import pytest


pa = pytest.importorskip("pyarrow")

from podium.datasets import Dataset, ExampleFactory
from podium.datasets.arrow import DiskBackedDataset
from podium.field import Field
from podium.vocab import Vocab


@pytest.fixture
def data():
    num_examples = 10
    numbers = range(num_examples)
    tokens = (
        " ".join(map(lambda x: str(x) + "abc", range(i - 1, i + 2)))
        for i in range(num_examples)
    )
    return list(zip(numbers, tokens))


@pytest.fixture
def fields():
    number_field = Field(
        "number", keep_raw=True, numericalizer=int, tokenizer=None, is_target=True
    )

    token_field = Field(
        "tokens",
        keep_raw=True,
        numericalizer=Vocab(keep_freqs=True),
        tokenizer=partial(str.split, sep=" "),
    )

    return [number_field, token_field]


@pytest.mark.usefixtures("data", "fields")
def pyarrow_dataset(data, fields):
    example_factory = ExampleFactory(fields)
    examples = map(example_factory.from_list, data)
    return DiskBackedDataset.from_examples(fields, examples)


@pytest.fixture(name="pyarrow_dataset")
def pyarrow_dataset_fixture(data, fields):
    ad = pyarrow_dataset(data, fields)
    yield ad
    ad.delete_cache()


def test_from_examples(data, fields):
    example_factory = ExampleFactory(fields)
    examples = [example_factory.from_list(ex) for ex in data]
    ad = DiskBackedDataset.from_examples(fields, examples)

    for (raw, tokenized), (num, _) in zip(ad.number, data):
        assert raw == num
        assert tokenized is num

    for (raw, tokenized), (_, tok) in zip(ad.tokens, data):
        assert raw == tok
        assert tokenized == tok.split(" ")

    ad.delete_cache()


@pytest.mark.parametrize(
    "index",
    [
        slice(0, 10),
        slice(None, None, 2),
        slice(None, None, -1),
        slice(2, 8, 2),
        (1, 5, 8),
        np.array([1, 5, 8]),
    ],
)
def test_slicing(index, data, fields, pyarrow_dataset):
    dataset_slice = pyarrow_dataset[index]

    if isinstance(index, slice):
        data_slice = data[index]
    else:
        data_slice = [data[i] for i in index]

    assert len(dataset_slice) == len(data_slice)
    for ex, (num, tok) in zip(dataset_slice, data_slice):
        num_raw, _ = ex["number"]
        tok_raw, _ = ex["tokens"]

        assert num_raw == num
        assert tok_raw == tok


def test_dump_and_load(pyarrow_dataset, tmpdir):
    cache_dir = pyarrow_dataset.dump_cache(cache_path=None)
    loaded_dataset = DiskBackedDataset.load_cache(cache_dir)

    assert len(loaded_dataset) == len(pyarrow_dataset)
    for ex_original, ex_loaded in zip(pyarrow_dataset, loaded_dataset):
        assert ex_original["number"] == ex_loaded["number"]
        assert ex_original["tokens"] == ex_loaded["tokens"]
    assert (
        pyarrow_dataset.field_dict["tokens"].vocab.stoi
        == loaded_dataset.field_dict["tokens"].vocab.stoi
    )

    loaded_dataset.delete_cache()

    dataset_sliced = pyarrow_dataset[8:2:-2]
    cache_dir_sliced = dataset_sliced.dump_cache(cache_path=None)
    loaded_dataset_sliced = DiskBackedDataset.load_cache(cache_dir_sliced)

    assert len(loaded_dataset_sliced) == len(dataset_sliced)
    for ex_original, ex_loaded in zip(dataset_sliced, loaded_dataset_sliced):
        assert ex_original["number"] == ex_loaded["number"]
        assert ex_original["tokens"] == ex_loaded["tokens"]
    assert (
        pyarrow_dataset.field_dict["tokens"].vocab.stoi
        == loaded_dataset_sliced.field_dict["tokens"].vocab.stoi
    )

    loaded_dataset_sliced.delete_cache()

    pyarrow_dataset.dump_cache(cache_path=tmpdir)
    loaded_dataset = DiskBackedDataset.load_cache(tmpdir)

    assert len(loaded_dataset) == len(pyarrow_dataset)
    for ex_original, ex_loaded in zip(pyarrow_dataset, loaded_dataset):
        assert ex_original["number"] == ex_loaded["number"]
        assert ex_original["tokens"] == ex_loaded["tokens"]
    assert (
        pyarrow_dataset.field_dict["tokens"].vocab.stoi
        == loaded_dataset.field_dict["tokens"].vocab.stoi
    )

    loaded_dataset.delete_cache()


def test_finalize_fields(data, fields, mocker):
    for field in fields:
        mocker.spy(field, "finalize")
        mocker.spy(field, "update_vocab")

    dataset = pyarrow_dataset(data, fields)

    for f in fields:
        # before finalization, no field's dict was updated
        if f.vocab is not None:
            assert not f.is_finalized

    dataset.finalize_fields()

    fields_to_finalize = [f for f in fields if not f.eager and f.use_vocab]
    for f in fields_to_finalize:
        # during finalization, only non-eager field's dict should be updated
        assert f.update_vocab.call_count == (len(data) if (not f.eager) else 0)
        f.finalize.assert_called_once()
        # all fields should be finalized
        assert f.is_finalized

    dataset.delete_cache()


def test_filtered(data, pyarrow_dataset):
    def filter_even(ex):
        return ex["number"][0] % 2 == 0

    filtered_dataset = pyarrow_dataset.filtered(filter_even)
    filtered_data = [d[0] for d in data if d[0] % 2 == 0]

    for (raw, _), d in zip(filtered_dataset.number, filtered_data):
        assert raw == d


def test_indexing(pyarrow_dataset, data):
    for i in range(len(data)):
        assert pyarrow_dataset[i].number[0] == data[i][0]


def test_batching(data, pyarrow_dataset):
    pyarrow_dataset.finalize_fields()
    batch = pyarrow_dataset.batch(add_padding=True)
    assert hasattr(batch, "number")
    assert hasattr(batch, "tokens")

    assert isinstance(batch.number, np.ndarray)
    assert len(batch.number) == len(pyarrow_dataset)
    for (raw, _), b in zip(pyarrow_dataset.number, batch.number):
        assert raw == b

    tokens_vocab = pyarrow_dataset.field_dict["tokens"].vocab
    for (_, tokenized), batch_row in zip(pyarrow_dataset.tokens, batch.tokens):
        assert len(tokenized) == len(batch_row)
        for token, index in zip(tokenized, batch_row):
            assert index == tokens_vocab[token]


def test_from_dataset(data, fields):
    example_factory = ExampleFactory(fields)
    examples = [example_factory.from_list(raw_example) for raw_example in data]
    dataset = Dataset(examples, fields)
    pyarrow_dataset = DiskBackedDataset.from_dataset(dataset)

    for ds_ex, arrow_ex in zip(dataset, pyarrow_dataset):
        assert ds_ex.number == arrow_ex.number
        assert ds_ex.tokens == arrow_ex.tokens

    pyarrow_dataset.delete_cache()


def test_from_tabular(data, fields, tmpdir):
    test_file = os.path.join(tmpdir, "test.csv")
    with open(test_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    csv_dataset = DiskBackedDataset.from_tabular_file(test_file, "csv", fields)
    for ex, d in zip(csv_dataset, data):
        assert int(ex.number[0]) == d[0]
        assert ex.tokens[0] == d[1]

    csv_dataset.delete_cache()


def test_missing_datatype_exception(data, fields, tmpdir):
    data_null = [(*d, None) for d in data]
    null_field = Field(
        "null_field", keep_raw=True, allow_missing_data=True, numericalizer=Vocab()
    )
    fields_null = [*fields, null_field]

    exf = ExampleFactory(fields_null)
    examples = map(exf.from_list, data_null)

    with pytest.raises(RuntimeError):
        DiskBackedDataset.from_examples(fields_null, examples, cache_path=tmpdir)


def test_datatype_definition(data, fields):
    data_null = [(*d, None) for d in data]
    null_field = Field(
        "null_field", keep_raw=True, allow_missing_data=True, numericalizer=Vocab()
    )
    fields_null = [*fields, null_field]

    exf = ExampleFactory(fields_null)
    examples = map(exf.from_list, data_null)

    datatypes = {"null_field": (pa.string(), pa.list_(pa.string()))}
    dataset = DiskBackedDataset.from_examples(fields_null, examples, data_types=datatypes)

    for ex, d in zip(dataset, data_null):
        assert int(ex["number"][0]) == d[0]
        assert ex["tokens"][0] == d[1]

    dataset.delete_cache()


def test_delete_cache(data, fields):
    cache_dir = tempfile.mkdtemp()

    example_factory = ExampleFactory(fields)
    examples = map(example_factory.from_list, data)
    ad = DiskBackedDataset.from_examples(fields, examples, cache_path=cache_dir)

    assert os.path.exists(cache_dir)
    ad.delete_cache()
    assert not os.path.exists(cache_dir)


def test_sorted(data, pyarrow_dataset):
    indices = [1, 5, 2, 7, 3]

    data_slice = [data[i] for i in indices]
    dataset_slice = pyarrow_dataset[indices]

    sorted_data = sorted(data_slice, key=lambda x: x[0], reverse=False)
    sorted_dataset = dataset_slice.sorted(key=lambda ex: ex["number"][0], reverse=False)
    for d, ex in zip(sorted_data, sorted_dataset):
        assert d[0] == ex["number"][0]

    reverse_sorted_data = sorted(data_slice, key=lambda x: x[0], reverse=True)
    reverse_sorted_dataset = dataset_slice.sorted(
        key=lambda ex: ex["number"][0], reverse=True
    )
    for d, ex in zip(reverse_sorted_data, reverse_sorted_dataset):
        assert d[0] == ex["number"][0]


def test_from_pandas_field_list(data):
    import pandas as pd

    df = pd.DataFrame(data)
    fields = [
        Field("number", tokenizer=None),
        Field("text", keep_raw=True, tokenizer="split"),
    ]

    ds = DiskBackedDataset.from_pandas(df, fields)

    for original, (raw, _) in zip(data, ds.text):
        assert original[1] == raw


def test_from_pandas_field_dict(data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["number", "text"])
    fields = {
        "number": Field("number", tokenizer=None),
        "text": Field("text", keep_raw=True, tokenizer="split"),
    }

    ds = DiskBackedDataset.from_pandas(df, fields)

    for original, (raw, _) in zip(data, ds.text):
        assert original[1] == raw


def test_from_pandas_index(data):
    import pandas as pd

    df = pd.DataFrame([[x[1]] for x in data], index=[x[0] for x in data])
    fields = [Field("text_field", keep_raw=True, tokenizer="split")]

    ds = DiskBackedDataset.from_pandas(
        df, fields, index_field=Field("number_field", tokenizer=None, keep_raw=True)
    )

    assert set(ds.field_dict) == set(["text_field", "number_field"])
    for original, (raw, _) in zip(data, ds.number_field):
        assert original[0] == raw
