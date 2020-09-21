import tempfile
from functools import partial

import numpy as np
import pytest

from podium.arrow import ArrowDataset
from podium.storage import Field, Vocab, ExampleFactory

# Try to import pyarrow
pa = pytest.importorskip('pyarrow')


@pytest.fixture
def pyarrow_data():
    num_examples = 10
    numbers = range(num_examples)
    tokens = ((" ".join(map(lambda x: str(x) + "abc", range(i - 1, i + 2))) for i in range(num_examples)))
    return list(zip(numbers, tokens))


@pytest.fixture
def pyarrow_fields():
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
def pyarrow_dataset(pyarrow_data, pyarrow_fields):
    example_factory = ExampleFactory(pyarrow_fields)
    examples = map(example_factory.from_list, pyarrow_data)
    return ArrowDataset.from_examples(pyarrow_fields, examples)


def test_from_examples(pyarrow_data, pyarrow_fields):
    example_factory = ExampleFactory(pyarrow_fields)
    examples = map(example_factory.from_list, iter(pyarrow_data))
    ad = ArrowDataset.from_examples(pyarrow_fields, examples)

    for (raw, tokenized), (num, _) in zip(ad.number, pyarrow_data):
        assert raw == num
        assert tokenized is None

    for (raw, tokenized), (_, tok) in zip(ad.tokens, pyarrow_data):
        assert raw == tok
        assert tokenized == tok.split(' ')


def test_slicing(pyarrow_data, pyarrow_fields, pyarrow_dataset):
    indices = [
        slice(0, len(pyarrow_data)),
        slice(None, None, 2),
        slice(None, None, -1),
        slice(2, 8, 2),
        (1, 5, 8),
        np.array([1, 5, 8]),
    ]

    for i in indices:
        dataset_slice = pyarrow_dataset[i]

        if isinstance(i, slice):
            data_slice = pyarrow_data[i]
        else:
            data_slice = [pyarrow_data[index] for index in i]

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
        assert ex_original == ex_loaded

    dataset_sliced = pyarrow_dataset[8:2:-2]
    cache_dir_sliced = dataset_sliced.dump_cache(cache_path=None)
    loaded_dataset_sliced = ArrowDataset.load_cache(cache_dir_sliced)

    assert len(loaded_dataset_sliced) == len(dataset_sliced)
    for ex_original, ex_loaded in zip(dataset_sliced, loaded_dataset_sliced):
        assert ex_original == ex_loaded

    cache_dir = tempfile.mkdtemp()
    pyarrow_dataset.dump_cache(cache_path=cache_dir)
    loaded_dataset = ArrowDataset.load_cache(cache_dir)

    assert len(loaded_dataset) == len(pyarrow_dataset)
    for ex_original, ex_loaded in zip(pyarrow_dataset, loaded_dataset):
        assert ex_original == ex_loaded


def test_finalize_fields(pyarrow_data, pyarrow_fields, mocker):
    for field in pyarrow_fields:
        mocker.spy(field, 'finalize')
        mocker.spy(field, 'update_vocab')

    dataset = pyarrow_dataset(pyarrow_data, pyarrow_fields)

    for f in pyarrow_fields:
        # before finalization, no field's dict was updated
        if f.vocab is not None:
            assert not f.finalized

    dataset.finalize_fields()

    fields_to_finalize = [f for f in pyarrow_fields if
                          not f.eager and f.use_vocab]
    for f in fields_to_finalize:
        # during finalization, only non-eager field's dict should be updated
        assert f.update_vocab.call_count == (len(pyarrow_data) if (not f.eager) else 0)
        f.finalize.assert_called_once()
        # all fields should be finalized
        assert f.finalized


def test_filter(pyarrow_data, pyarrow_dataset):
    def filter_even(ex):
        return ex.number[0] % 2 == 0

    filtered_dataset = pyarrow_dataset.filter(filter_even)
    filtered_data = [d[0] for d in pyarrow_data if d[0] % 2 == 0]

    for (raw, _), d in zip(filtered_dataset.number, filtered_data):
        assert raw == d


def test_indexing(pyarrow_dataset, pyarrow_data):
    for i in range(len(pyarrow_data)):
        assert pyarrow_dataset[i].number[0] == pyarrow_data[i][0]


def test_batching(pyarrow_data, pyarrow_dataset):
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
