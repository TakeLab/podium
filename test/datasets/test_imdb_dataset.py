import os
import tempfile
import pytest
from takepod.datasets.impl.imdb_sentiment_dataset import BasicSupervisedImdbDataset
from takepod.datasets.dataset import Dataset
from takepod.storage.resources.large_resource import LargeResource


TRAIN_EXAMPLES = {
    "pos": ["If you like comedy cartoons then this is nearly a similar format",
            "I came in in the middle of this film so I had no idea about any credit",
            "The production quality, cast, premise, authentic New England"],
    "neg": ["If I had not read Pat Barker's 'Union Street' before seeing this film",
            "There are lots of extremely good-looking people"]
}

EXPECTED_TRAIN_EXAMPLES = [
    {BasicSupervisedImdbDataset.TEXT_FIELD_NAME: "If you like comedy cartoons then "
                                                 "this is nearly a similar "
                                                 "format".split(),
     BasicSupervisedImdbDataset.LABEL_FIELD_NAME: 1},
    {BasicSupervisedImdbDataset.TEXT_FIELD_NAME: "I came in in the middle of this "
                                                 "film so I had no idea about any "
                                                 "credit".split(),
     BasicSupervisedImdbDataset.LABEL_FIELD_NAME: 1},
    {BasicSupervisedImdbDataset.TEXT_FIELD_NAME: "The production quality, cast, "
                                                 "premise, authentic New "
                                                 "England".split(),
     BasicSupervisedImdbDataset.LABEL_FIELD_NAME: 1},
    {BasicSupervisedImdbDataset.TEXT_FIELD_NAME: "If I had not read Pat Barker's "
                                                 "'Union Street' before seeing this "
                                                 "film".split(),
     BasicSupervisedImdbDataset.LABEL_FIELD_NAME: 0},
    {BasicSupervisedImdbDataset.TEXT_FIELD_NAME: "There are lots of extremely "
                                                 "good-looking people".split(),
     BasicSupervisedImdbDataset.LABEL_FIELD_NAME: 0}
]

TEST_EXAMPLES = {
    "pos": ["My yardstick for measuring a movie's watch-ability is if I get squirmy."
            "The Dresser is perhaps the most refined of backstage films"],
    "neg": ["It seems ever since 1982, about every two or three years"]
}


@pytest.fixture(scope="module")
def mock_dataset_path():
    base_temp = tempfile.mkdtemp()
    assert os.path.exists(base_temp)
    LargeResource.BASE_RESOURCE_DIR = base_temp
    base_dataset_dir = os.path.join(base_temp, "imdb",
                                    "aclImdb")

    train_dir = os.path.join(base_dataset_dir, "train")
    os.makedirs(train_dir)
    assert os.path.exists(train_dir)
    test_dir = os.path.join(base_dataset_dir, "test")
    os.makedirs(test_dir)
    assert os.path.exists(test_dir)

    create_examples_set(train_dir, TRAIN_EXAMPLES)
    create_examples_set(test_dir, TEST_EXAMPLES)
    return base_temp


def create_examples_set(base_dir, examples):
    pos_dir = os.path.join(base_dir, 'pos')
    os.makedirs(pos_dir)
    assert os.path.exists(pos_dir)
    neg_dir = os.path.join(base_dir, 'neg')
    os.makedirs(neg_dir)
    assert os.path.exists(neg_dir)

    create_examples(pos_dir, examples['pos'])
    create_examples(neg_dir, examples['neg'])


def create_examples(base_dir, examples):
    for i in range(len(examples)):
        file_name = f"{i}_1.txt"
        with open(file=os.path.join(base_dir, file_name),
                  mode='w', encoding="utf8") as fpr:
            fpr.write(examples[i])


def test_return_params(mock_dataset_path):
    data = BasicSupervisedImdbDataset.get_train_test_dataset()
    assert len(data) == 2
    assert isinstance(data[0], Dataset)
    assert isinstance(data[1], Dataset)


def test_default_fields():
    fields = BasicSupervisedImdbDataset.get_default_fields()
    assert len(fields) == 2
    field_names = [BasicSupervisedImdbDataset.LABEL_FIELD_NAME,
                   BasicSupervisedImdbDataset.TEXT_FIELD_NAME]
    assert all([name in fields for name in field_names])


def test_loaded_data(mock_dataset_path):
    data = BasicSupervisedImdbDataset.get_train_test_dataset()
    train_dataset, _ = data
    for ex in train_dataset:
        ex_data = {"text": ex.text[1], "label": ex.label[0]}
        assert ex_data in EXPECTED_TRAIN_EXAMPLES
