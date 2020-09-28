import tempfile
from functools import partial
import numpy as np
import pytest
import os
import csv

from podium.arrow import ArrowDataset
from podium.storage import Field, Vocab, ExampleFactory
from podium.datasets import Dataset

# Try to import pyarrow
pa = pytest.importorskip('pyarrow')


@pytest.fixture
def data():
    num_examples = 10
    numbers = range(num_examples)
    tokens = ((" ".join(map(lambda x: str(x) + "abc", range(i - 1, i + 2)))
               for i in range(num_examples))
              )
    return list(zip(numbers, tokens))


@pytest.fixture
def fields():
    number_field = Field('number',
                         store_as_raw=True,
                         custom_numericalize=int,
                         tokenize=False,
                         is_target=True)

    token_field = Field('tokens',
                        store_as_raw=True,
                        vocab=Vocab(keep_freqs=True),
                        tokenizer=partial(str.split, sep=' '))

    return [number_field, token_field]


@pytest.fixture()
def pyarrow_dataset(data, fields):
    example_factory = ExampleFactory(fields)
    examples = map(example_factory.from_list, data)
    return ArrowDataset.from_examples(fields, examples)


def test_from_examples(data, fields):
    example_factory = ExampleFactory(fields)
    examples = map(example_factory.from_list, iter(data))
    ad = ArrowDataset.from_examples(fields, examples)

    for (raw, tokenized), (num, _) in zip(ad.number, data):
        assert raw == num
        assert tokenized is None

    for (raw, tokenized), (_, tok) in zip(ad.tokens, data):
        assert raw == tok
        assert tokenized == tok.split(' ')


def test_slicing(data, fields, pyarrow_dataset):
    indices = [
        slice(0, len(data)),
        slice(None, None, 2),
        slice(None, None, -1),
        slice(2, 8, 2),
        (1, 5, 8),
        np.array([1, 5, 8]),
    ]

    for i in indices:
        dataset_slice = pyarrow_dataset[i]

        if isinstance(i, slice):
            data_slice = data[i]
        else:
            data_slice = [data[index] for index in i]

        assert len(dataset_slice) == len(data_slice)
        for ex, (num, tok) in zip(dataset_slice, data_slice):
            num_raw, _ = getattr(ex, 'number')
            tok_raw, _ = getattr(ex, 'tokens')

            assert num_raw == num
            assert tok_raw == tok


def test_dump_and_load(pyarrow_dataset):
    cache_dir = pyarrow_dataset.dump_cache(cache_path=None)
    loaded_dataset = ArrowDataset.load_cache(cache_dir)

    assert len(loaded_dataset) == len(pyarrow_dataset)
    for ex_original, ex_loaded in zip(pyarrow_dataset, loaded_dataset):
        assert ex_original.number == ex_loaded.number
        assert ex_original.tokens == ex_loaded.tokens

    dataset_sliced = pyarrow_dataset[8:2:-2]
    cache_dir_sliced = dataset_sliced.dump_cache(cache_path=None)
    loaded_dataset_sliced = ArrowDataset.load_cache(cache_dir_sliced)

    assert len(loaded_dataset_sliced) == len(dataset_sliced)
    for ex_original, ex_loaded in zip(dataset_sliced, loaded_dataset_sliced):
        assert ex_original.number == ex_loaded.number
        assert ex_original.tokens == ex_loaded.tokens

    cache_dir = tempfile.mkdtemp()
    pyarrow_dataset.dump_cache(cache_path=cache_dir)
    loaded_dataset = ArrowDataset.load_cache(cache_dir)

    assert len(loaded_dataset) == len(pyarrow_dataset)
    for ex_original, ex_loaded in zip(pyarrow_dataset, loaded_dataset):
        assert ex_original.number == ex_loaded.number
        assert ex_original.tokens == ex_loaded.tokens


def test_finalize_fields(data, fields, mocker):
    for field in fields:
        mocker.spy(field, 'finalize')
        mocker.spy(field, 'update_vocab')

    dataset = pyarrow_dataset(data, fields)

    for f in fields:
        # before finalization, no field's dict was updated
        if f.vocab is not None:
            assert not f.finalized

    dataset.finalize_fields()

    fields_to_finalize = [f for f in fields if
                          not f.eager and f.use_vocab]
    for f in fields_to_finalize:
        # during finalization, only non-eager field's dict should be updated
        assert f.update_vocab.call_count == (len(data) if (not f.eager) else 0)
        f.finalize.assert_called_once()
        # all fields should be finalized
        assert f.finalized


def test_filter(data, pyarrow_dataset):
    def filter_even(ex):
        return ex.number[0] % 2 == 0

    filtered_dataset = pyarrow_dataset.filter(filter_even)
    filtered_data = [d[0] for d in data if d[0] % 2 == 0]

    for (raw, _), d in zip(filtered_dataset.number, filtered_data):
        assert raw == d


def test_indexing(pyarrow_dataset, data):
    for i in range(len(data)):
        assert pyarrow_dataset[i].number[0] == data[i][0]


def test_batching(data, pyarrow_dataset):
    pyarrow_dataset.finalize_fields()
    input_batch, target_batch = pyarrow_dataset.batch()
    assert hasattr(target_batch, 'number')
    assert hasattr(input_batch, 'tokens')

    assert isinstance(target_batch.number, np.ndarray)
    assert len(target_batch.number) == len(pyarrow_dataset)
    for (raw, _), b in zip(pyarrow_dataset.number, target_batch.number):
        assert raw == b

    tokens_vocab = pyarrow_dataset.field_dict['tokens'].vocab
    for (_, tokenized), batch_row in zip(pyarrow_dataset.tokens, input_batch.tokens):
        assert len(tokenized) == len(batch_row)
        for token, index in zip(tokenized, batch_row):
            assert index == tokens_vocab[token]


def test_to_dataset(pyarrow_dataset):
    dataset = pyarrow_dataset.as_dataset()

    assert isinstance(dataset, Dataset)
    assert len(dataset) == len(pyarrow_dataset)
    for arrow_ex, dataset_ex in zip(pyarrow_dataset, dataset):
        assert arrow_ex.number == dataset_ex.number
        assert arrow_ex.tokens == dataset_ex.tokens


def test_from_dataset(data, fields):
    example_factory = ExampleFactory(fields)
    examples = list(map(example_factory.from_list, data))
    dataset = Dataset(examples, fields)
    pyarrow_dataset = ArrowDataset.from_dataset(dataset)

    for ds_ex, arrow_ex in zip(dataset, pyarrow_dataset):
        assert ds_ex.number == arrow_ex.number
        assert ds_ex.tokens == arrow_ex.tokens


def test_from_tabular(data, fields):
    with tempfile.TemporaryDirectory() as tdir:
        test_file = os.path.join(tdir, 'test.csv')
        with open(test_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        csv_dataset = ArrowDataset.from_tabular_file(test_file, 'csv', fields)
        for ex, d in zip(csv_dataset, data):
            assert int(ex.number[0]) == d[0]
            assert ex.tokens[0] == d[1]


def test_missing_datatype_exception(data, fields):
    data_null = [(*d, None) for d in data]
    null_field = Field('null_field',
                       store_as_raw=True,
                       allow_missing_data=True,
                       vocab=Vocab())
    fields_null = [*fields, null_field]

    exf = ExampleFactory(fields_null)
    examples = map(exf.from_list, data_null)

    with pytest.raises(Exception):
        ArrowDataset.from_examples(fields_null, examples)


def test_datatype_definition(data, fields):
    data_null = [(*d, None) for d in data]
    null_field = Field('null_field',
                       store_as_raw=True,
                       allow_missing_data=True,
                       vocab=Vocab())
    fields_null = [*fields, null_field]

    exf = ExampleFactory(fields_null)
    examples = map(exf.from_list, data_null)

    datatypes = {'null_field': (pa.string(), pa.list_(pa.string()))}
    dataset = ArrowDataset.from_examples(fields_null, examples, data_types=datatypes)

    for ex, d in zip(dataset, data_null):
        assert int(ex.number[0]) == d[0]
        assert ex.tokens[0] == d[1]


def test_delete_cache(data, fields):
    cache_dir = tempfile.mkdtemp()

    example_factory = ExampleFactory(fields)
    examples = map(example_factory.from_list, iter(data))
    ad = ArrowDataset.from_examples(fields, examples, cache_path=cache_dir)

    assert os.path.exists(cache_dir)
    ad.delete_cache()
    assert not os.path.exists(cache_dir)