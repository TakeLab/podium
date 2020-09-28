import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

import pytest

from podium.datasets.dataset import Dataset
from podium.datasets.impl.croatian_ner_dataset import CroatianNERDataset
from podium.storage.resources.large_resource import LargeResource


title_1 = ET.fromstring(
    """
<title>
    <s>
        <enamex type="LocationAsOrganization">Kina</enamex>
        je najveći svjetski izvoznik
    </s>
</title>
"""
)

body_1 = ET.fromstring(
    """
<body>
    <s>
        Ukupna vrijednost izvoza <timex type="Date">u prvoj polovini
        ove godine</timex> iznosila je <numex type="Money">521,7
        milijardi dolara</numex>.
    </s>
</body>
"""
)


def create_ner_file(filepath, title_element, body_element):
    root = ET.Element("root")

    root.append(title_element)
    root.append(body_element)

    tree = ET.ElementTree(root)
    tree.write(filepath, encoding="utf-8")


@pytest.fixture(scope="function")
def base_download_dir():
    base_temp = tempfile.mkdtemp()
    assert os.path.isdir(base_temp)

    base_dataset_dir = os.path.join(base_temp, "CroatianNERDataset")
    os.makedirs(base_dataset_dir)

    ner_file_name = os.path.join(base_dataset_dir, "example.xml")
    create_ner_file(ner_file_name, title_1, body_1)

    return base_temp


def test_return_params(base_download_dir):
    LargeResource.BASE_RESOURCE_DIR = base_download_dir

    dataset = CroatianNERDataset.get_dataset()
    shutil.rmtree(base_download_dir)

    assert isinstance(dataset, Dataset)


def test_default_fields():
    fields = CroatianNERDataset.get_default_fields()
    assert len(fields) == 2
    field_names = {"tokens", "labels"}
    assert all([name in field_names for name in fields])


def test_dataset_loading(base_download_dir):
    LargeResource.BASE_RESOURCE_DIR = base_download_dir
    dataset = CroatianNERDataset.get_dataset()
    shutil.rmtree(base_download_dir)

    assert len(dataset) == 2

    expected_tokens_0 = ["Kina", "je", "najveći", "svjetski", "izvoznik"]
    expected_labels_0 = ["B-LocationAsOrganization", "O", "O", "O", "O"]
    assert dataset[0].tokens[1] == expected_tokens_0
    assert dataset[0].labels[1] == expected_labels_0

    expected_tokens_1 = [
        "Ukupna",
        "vrijednost",
        "izvoza",
        "u",
        "prvoj",
        "polovini",
        "ove",
        "godine",
        "iznosila",
        "je",
        "521,7",
        "milijardi",
        "dolara",
        ".",
    ]
    expected_labels_1 = [
        "O",
        "O",
        "O",
        "B-Date",
        "I-Date",
        "I-Date",
        "I-Date",
        "I-Date",
        "O",
        "O",
        "B-Money",
        "I-Money",
        "I-Money",
        "O",
    ]
    assert dataset[1].tokens[1] == expected_tokens_1
    assert dataset[1].labels[1] == expected_labels_1
