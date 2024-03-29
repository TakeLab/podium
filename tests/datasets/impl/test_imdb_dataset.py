import os
import tempfile

import pytest

from podium.datasets.dataset import Dataset
from podium.datasets.impl.imdb import IMDB
from podium.storage.resources.large_resource import LargeResource
from podium.utils.general_utils import load_spacy_model_or_raise


TRAIN_EXAMPLES = {
    "pos": [
        "If you like comedy cartoons then this is nearly a similar format",
        "I came in in the middle of this film so I had no idea about any credit",
        "The production quality, cast, premise, authentic New England",
    ],
    "neg": [
        "If I had not read Pat Barker's 'Union Street' before seeing this film",
        "There are lots of extremely good-looking people",
    ],
}

EXPECTED_TRAIN_EXAMPLES = [
    {
        "text": "If you like comedy cartoons then this is nearly a similar format",
        "label": "positive",
    },
    {
        "text": "I came in in the middle of this "
        "film so I had no idea about any "
        "credit",
        "label": "positive",
    },
    {
        "text": "The production quality, cast, premise, authentic New England",
        "label": "positive",
    },
    {
        "text": "If I had not read Pat Barker's "
        "'Union Street' before seeing this "
        "film",
        "label": "negative",
    },
    {"text": "There are lots of extremely good-looking people", "label": "negative"},
]

TEST_EXAMPLES = {
    "pos": [
        "My yardstick for measuring a movie's watch-ability is if I get squirmy."
        "The Dresser is perhaps the most refined of backstage films"
    ],
    "neg": ["It seems ever since 1982, about every two or three years"],
}


@pytest.fixture(scope="module", autouse=True)
def mock_dataset_path():
    with tempfile.TemporaryDirectory() as base_temp:
        assert os.path.exists(base_temp)
        LargeResource.BASE_RESOURCE_DIR = base_temp
        base_dataset_dir = os.path.join(base_temp, "imdb", "aclImdb")

        train_dir = os.path.join(base_dataset_dir, "train")
        os.makedirs(train_dir)
        assert os.path.exists(train_dir)
        test_dir = os.path.join(base_dataset_dir, "test")
        os.makedirs(test_dir)
        assert os.path.exists(test_dir)

        create_examples_set(train_dir, TRAIN_EXAMPLES)
        create_examples_set(test_dir, TEST_EXAMPLES)

        yield base_temp


def create_examples_set(base_dir, examples):
    pos_dir = os.path.join(base_dir, "pos")
    os.makedirs(pos_dir)
    assert os.path.exists(pos_dir)
    neg_dir = os.path.join(base_dir, "neg")
    os.makedirs(neg_dir)
    assert os.path.exists(neg_dir)

    create_examples(pos_dir, examples["pos"])
    create_examples(neg_dir, examples["neg"])


def create_examples(base_dir, examples):
    for i in range(len(examples)):
        file_name = f"{i}_1.txt"
        with open(
            file=os.path.join(base_dir, file_name), mode="w", encoding="utf8"
        ) as fpr:
            fpr.write(examples[i])


@pytest.mark.require_package("spacy")
@pytest.mark.require_spacy_model("en_core_web_sm")
def test_return_params(mock_dataset_path):
    data = IMDB.get_dataset_splits()
    assert len(data) == 2
    assert isinstance(data[0], Dataset)
    assert isinstance(data[1], Dataset)


@pytest.mark.require_package("spacy")
@pytest.mark.require_spacy_model("en_core_web_sm")
def test_default_fields():
    fields = IMDB.get_default_fields()
    assert len(fields) == 2
    field_names = [IMDB.LABEL_FIELD_NAME, IMDB.TEXT_FIELD_NAME]
    assert all([name in fields for name in field_names])


@pytest.mark.require_package("spacy")
@pytest.mark.require_spacy_model("en_core_web_sm")
def test_loaded_data(mock_dataset_path):
    spacy_tokenizer = load_spacy_model_or_raise(
        "en_core_web_sm", disable=["parser", "ner"]
    )

    def spacy_tokenize(string):
        return [token.text for token in spacy_tokenizer.tokenizer(string)]

    expected_data = []
    for expected_example in EXPECTED_TRAIN_EXAMPLES:
        expected_data.append(
            {
                "text": spacy_tokenize(expected_example["text"]),
                "label": expected_example["label"],
            }
        )

    data = IMDB.get_dataset_splits()
    train_dataset, _ = data
    assert len(train_dataset) > 0

    for ex in train_dataset:
        real_example_data = {"text": ex["text"][1], "label": ex["label"][1]}
        assert real_example_data in expected_data
