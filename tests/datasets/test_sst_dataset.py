import os
import tempfile

import pytest

from podium.datasets.dataset import Dataset
from podium.datasets.impl.sst_sentiment_dataset import SST
from podium.storage.resources.large_resource import LargeResource


EXPECTED_TRAIN_EXAMPLES = [
    {
        "text": (
            "The Rock is destined to be the 21st Century 's new `` Conan ''"
            " and that he 's going to make a splash even greater than "
            "Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
        ).split(),
        "label": "positive",
    },
    {"text": "This is n't a new idea .".split(), "label": "negative"},
    {"text": "This is n't a bad idea .".split(), "label": "neutral"},
]

RAW_EXAMPLES = [
    "(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) "
    "(2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's)) "
    "(2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 '')) (2 and)) (3 (2 that) (3 (2 he) "
    "(3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) "
    "(2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) "
    "(2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) "
    "(2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .))) ",
    "(1 (2 This) (2 (1 (1 (2 is) (2 n't)) (3 (2 a) (2 (3 new) (2 idea)))) (2 .)))",
    "(2 (2 This) (2 (1 (1 (2 is) (2 n't)) (3 (2 a) (2 (1 bad) (2 idea)))) (2 .)))",
]


@pytest.fixture(scope="module")
def mock_dataset_path():
    base_temp = tempfile.mkdtemp()
    assert os.path.exists(base_temp)
    LargeResource.BASE_RESOURCE_DIR = base_temp
    base_dataset_dir = os.path.join(base_temp, "trees")
    os.makedirs(base_dataset_dir)
    train_filename = os.path.join(base_dataset_dir, "train.txt")
    create_examples(train_filename, RAW_EXAMPLES)

    return train_filename


def create_examples(file_name, raw_examples):
    with open(file=file_name, mode="w", encoding="utf8") as fpr:
        for example in raw_examples:
            fpr.write(example + "\n")


def test_default_fields():
    fields = SST.get_default_fields()
    assert len(fields) == 2
    field_names = [SST.TEXT_FIELD_NAME, SST.LABEL_FIELD_NAME]
    assert all([name in fields for name in field_names])
    # Label field is a target
    assert fields[SST.LABEL_FIELD_NAME].is_target
    # Text field is not a target
    assert not fields[SST.TEXT_FIELD_NAME].is_target


def test_load_dataset(mock_dataset_path):
    train_dataset = SST(file_path=mock_dataset_path, fields=SST.get_default_fields())
    train_dataset.finalize_fields()
    assert isinstance(train_dataset, Dataset)

    assert len(train_dataset) == 2
    assert len(train_dataset.fields[1].vocab) == 2
    # The neutral example will be filtered
    for ex in train_dataset:
        ex_data = {"text": ex.text[1], "label": ex.label[0]}
        assert ex_data in EXPECTED_TRAIN_EXAMPLES[:2]


def test_load_finegrained(mock_dataset_path):
    train_dataset = SST(
        file_path=mock_dataset_path, fields=SST.get_default_fields(), fine_grained=True
    )
    train_dataset.finalize_fields()
    assert isinstance(train_dataset, Dataset)

    assert len(train_dataset) == 3
    assert len(train_dataset.fields[1].vocab) == 3
    for ex in train_dataset:
        ex_data = {"text": ex.text[1], "label": ex.label[0]}
        assert ex_data in EXPECTED_TRAIN_EXAMPLES


def test_load_subtrees(mock_dataset_path):
    train_dataset = SST(
        file_path=mock_dataset_path, fields=SST.get_default_fields(), subtrees=True
    )
    train_dataset.finalize_fields()
    assert isinstance(train_dataset, Dataset)

    # The length of all subtrees is 26
    assert len(train_dataset) == 26


def test_load_subtrees_finegrained(mock_dataset_path):
    train_dataset = SST(
        file_path=mock_dataset_path,
        fields=SST.get_default_fields(),
        subtrees=True,
        fine_grained=True,
    )
    train_dataset.finalize_fields()
    assert isinstance(train_dataset, Dataset)

    # The length of all subtrees is 97
    assert len(train_dataset) == 97
