import os
import tempfile
import pytest
from takepod.datasets.eurovoc_dataset import EuroVocDataset

EXAMPLES = r"""{
        "Zakon 1": {
        "title": "Zakon 1",
        "text": "Tekst zakona 1.",
        "labels": [
            1
        ],
        "filename": "file_1.xml"
        },
        "Zakon 2": {
        "title": "Zakon 2",
        "text": "Tekst zakona 2.",
        "labels": [
            1,
            2
        ],
        "filename": "file_2.xml"
        }
        }
        """

EXPECTED_EXAMPLES = [
    {"title": r"Zakon 1".split(),
     "text": r"Tekst zakona 1.".split(),
     "labels": [1]},
    {"title": r"Zakon 2".split(),
     "text": r"Tekst zakona 2.".split(),
     "labels": [1, 2]}]

LABELS = r"""{
        "1": {
            "similar_terms": [],
            "id": 1,
            "name": "politika",
            "rank": "thezaurus",
            "micro_thezaurus": null,
            "parents": [],
            "thezaurus": 1
        },
        "2": {
            "similar_terms": [],
            "id": 2,
            "name": "politički okvir",
            "rank": "micro_thezaurus",
            "micro_thezaurus": 2,
            "parents": [
                4
            ],
            "thezaurus": 4
        }
        }
        """

EXPECTED_LABELS = [
    {"id": 1,
     "name": "politika",
     "rank": "thezaurus",
     "micro_thezaurus": None,
     "thezaurus": 1,
     "parents": [],
     "similar_terms": []},
    {"id": 2,
     "name": "politički okvir",
     "rank": "micro_thezaurus",
     "micro_thezaurus": 2,
     "thezaurus": 4,
     "parents": [4],
     "similar_terms": []},
]


@pytest.fixture(scope="module")
def mock_dataset_path():
    base_temp = tempfile.mkdtemp()
    assert os.path.exists(base_temp)
    dataset_path = os.path.join(base_temp, "dataset")
    labels_path = os.path.join(base_temp, "labels")

    create_file(dataset_path, EXAMPLES)
    create_file(labels_path, LABELS)
    return dataset_path, labels_path


def create_file(file_path, file_content):
    with open(file=file_path, mode='w', encoding="utf8") as fp:
        fp.writelines(file_content)


def test_default_fields():
    fields = EuroVocDataset._get_default_fields()
    assert len(fields) == 3
    field_names = ["text", "title", "labels"]
    assert all([name in fields for name in field_names])


def test_loading_dataset(mock_dataset_path):
    dataset_path, labels_path = mock_dataset_path
    data = EuroVocDataset(dataset_path=dataset_path, label_hierarchy_path=labels_path)
    assert len(data.examples) == 2


def test_loaded_data(mock_dataset_path):
    dataset_path, labels_path = mock_dataset_path
    data = EuroVocDataset(dataset_path=dataset_path, label_hierarchy_path=labels_path)

    for ex in data:
        ex_data = {"title": ex.title[1], "text": ex.text[1], "labels": ex.labels[0]}
        assert ex_data in EXPECTED_EXAMPLES


def test_loading_labels(mock_dataset_path):
    dataset_path, labels_path = mock_dataset_path
    data = EuroVocDataset(dataset_path=dataset_path, label_hierarchy_path=labels_path)
    assert len(data.get_label_hierarchy()) == 2


def test_loaded_labels(mock_dataset_path):
    dataset_path, labels_path = mock_dataset_path
    data = EuroVocDataset(dataset_path=dataset_path, label_hierarchy_path=labels_path)
    labels = data.get_label_hierarchy()

    for label_id in labels:
        label = labels[label_id]
        label_data = {"id": label.id, "name": label.name, "rank": label.rank,
                      "micro_thezaurus": label.micro_thezaurus,
                      "thezaurus": label.thezaurus, "parents": label.parents,
                      "similar_terms": label.similar_terms}
        assert label_data in EXPECTED_LABELS
