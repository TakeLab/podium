from podium.arrow import ArrowDataset
from podium.datasets import Dataset
from podium.storage import Field, Vocab, ExampleFactory

import pytest
from functools import partial

# Try to import pyarrow
pa = pytest.importorskip('pyarrow')


@pytest.fixture
def data():
    num_examples = 10
    numbers = range(num_examples)
    tokens = ((" ".join(map(lambda x: str(x) + "abc", range(i - 1, i + 2))) for i in range(num_examples)))
    return zip(numbers, tokens)


@pytest.fixture
def fields():
    number_field = Field('number',
                         store_as_raw=True,
                         tokenizer=int,
                         is_target=True)

    token_field = Field('tokens',
                        store_as_raw=True,
                        vocab=Vocab(keep_freqs=True),
                        tokenizer=partial(str.split, sep=' '))

    return [number_field, token_field]


def test_from_examples(data, fields):
    example_factory = ExampleFactory(fields)
    examples = map(example_factory.from_list, data)

    ads = ArrowDataset.from_examples(fields, examples)
