import os
import tempfile
import pytest
from podium.datasets.impl.snli_dataset import SNLISimple
from podium.datasets.dataset import Dataset
from podium.storage.resources.large_resource import LargeResource


# TODO: Add full test examples once LabelField is merged to master.

TRAIN_EXAMPLES_SIMPLE = [
    '{"gold_label": "neutral", "sentence1": "Lorem ipsum '
    'dolor sit amet, consectetur adipiscing elit.", '
    '"sentence2": "Vestibulum vehicula, eros at bibendum '
    'convallis, nulla eros luctus dui."}',
    '{"gold_label": "contradiction", "sentence1": "Fusce '
    'vehicula suscipit diam eu posuere. Aliquam ante diam.", '
    '"sentence2": "Mauris eros libero, mattis quis odio ut, '
    'dictum consequat lorem."}'
]

EXPECTED_EXAMPLES_SIMPLE = [
    {"gold_label": "neutral",
     "sentence1": ("Lorem ipsum dolor sit amet, "
                   "consectetur adipiscing elit.").split(),
     "sentence2": ("Vestibulum vehicula, eros at bibendum "
                   "convallis, nulla eros luctus dui.").split()},
    {"gold_label": "contradiction",
     "sentence1": ("Fusce vehicula suscipit diam eu "
                   "posuere. Aliquam ante diam.").split(),
     "sentence2": ("Mauris eros libero, mattis quis "
                   "odio ut, dictum consequat lorem.").split()}
]


@pytest.fixture(scope="module")
def mock_dataset_path():
    base_temp = tempfile.mkdtemp()
    assert os.path.exists(base_temp)
    LargeResource.BASE_RESOURCE_DIR = base_temp
    base_dataset_dir = os.path.join(base_temp, "snli_1.0")
    os.makedirs(base_dataset_dir)
    dataset_simple = os.path.join(base_dataset_dir, "simple.jsonl")
    create_examples(dataset_simple, TRAIN_EXAMPLES_SIMPLE)
    return dataset_simple


def create_examples(file_name, raw_examples):
    with open(file=file_name, mode='w', encoding="utf8") as fpr:
        for example in raw_examples:
            fpr.write(example + "\n")


def test_default_fields():
    fields = SNLISimple.get_default_fields()
    assert len(fields) == 3
    field_names = [SNLISimple.GOLD_LABEL_FIELD_NAME,
                   SNLISimple.SENTENCE1_FIELD_NAME,
                   SNLISimple.SENTENCE2_FIELD_NAME]
    assert all([name in fields for name in field_names])
    # Label field is a target
    assert fields[SNLISimple.GOLD_LABEL_FIELD_NAME].is_target
    # Text field is not a target
    assert not fields[SNLISimple.SENTENCE1_FIELD_NAME].is_target
    assert not fields[SNLISimple.SENTENCE2_FIELD_NAME].is_target


def test_load_dataset(mock_dataset_path):
    dataset = SNLISimple(mock_dataset_path, fields=SNLISimple.get_default_fields())
    assert isinstance(dataset, Dataset)

    assert len(dataset) == 2
    for ex in dataset:
        ex_data = {"gold_label": ex.gold_label[0],
                   "sentence1": ex.sentence1[1],
                   "sentence2": ex.sentence2[1]}
        assert ex_data in EXPECTED_EXAMPLES_SIMPLE
