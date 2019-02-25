import xml.etree.ElementTree as ET
import tempfile
import os
import shutil
import pytest

from takepod.dataload.pauzahr import PauzaHR


def create_review_file(filepath, review_text, review_rating):
    root = ET.Element("root")
    text = ET.SubElement(root, "Text")
    text.text = review_text
    source = ET.SubElement(root, "Source")
    source.text = "quality source"
    rating = ET.SubElement(root, "Rating")
    rating.text = review_rating

    tree = ET.ElementTree(root)
    tree.write(filepath)


@pytest.mark.parametrize(
    "is_train, expected_data, expected_labels",
    [
        (True, [('good review', 'quality source')], [5.2]),
        (False, [('test review', 'quality source')], [2.3])
    ]
)
def test_load_dataset(tmpdir, is_train, expected_data, expected_labels):
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    unzipped = os.path.join(
        base, "croopinion", "CropinionDataset", "reviews_original"
    )
    # create all directories
    os.makedirs(unzipped)
    os.makedirs(os.path.join(unzipped, "Train"))
    os.makedirs(os.path.join(unzipped, "Test"))

    create_review_file(
        os.path.join(unzipped, "Train", "file.xml"), "good review", '5.2'
    )
    create_review_file(
        os.path.join(unzipped, "Test", "file.xml"), "test review", '2.3'
    )

    pauzaDs = PauzaHR()
    pauzaDs._data_dir = base
    x, y = pauzaDs.load_data(train=is_train)
    assert x == expected_data
    assert y == expected_labels

    shutil.rmtree(base)
    assert not os.path.exists(base)
