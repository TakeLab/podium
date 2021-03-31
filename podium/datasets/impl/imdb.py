"""
Module contains IMDB Large Movie Review Dataset Dataset webpage:
http://ai.stanford.edu/~amaas/data/sentiment/

When using this dataset, please cite:
    @InProceedings{maas-EtAl:2011:ACL-HLT2011,
    author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and
    Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
    title     = {Learning Word Vectors for Sentiment Analysis},
    booktitle = {Proceedings of the 49th Annual Meeting of the Association for
    Computational Linguistics: Human Language Technologies},
    month     = {June},
    year      = {2011},
    address   = {Portland, Oregon, USA},
    publisher = {Association for Computational Linguistics},
    pages     = {142--150},
    url       = {http://www.aclweb.org/anthology/P11-1015}
    }
"""

import os

from podium.datasets.dataset import Dataset
from podium.datasets.example_factory import ExampleFactory
from podium.field import Field, LabelField
from podium.storage.resources.large_resource import LargeResource
from podium.vocab import Vocab


class IMDB(Dataset):
    """
    Simple Imdb dataset with only supervised data which uses non processed data.

    Attributes
    ----------
    NAME : str
        dataset name
    URL : str
        url to the imdb dataset
    DATASET_DIR : str
        name of the folder in the dataset containing train and test directories
    ARCHIVE_TYPE : str
        string that defines archive type, used for unpacking dataset
    TRAIN_DIR : str
        name of the training directory
    TEST_DIR : str
        name of the directory containing test examples
    POSITIVE_LABEL_DIR : str
        name of the subdirectory containing examples with positive sentiment
    NEGATIVE_LABEL_DIR : str
        name of the subdirectory containing examples with negative sentiment
    TEXT_FIELD_NAME : str
        name of the field containing comment text
    LABEL_FIELD_NAME : str
        name of the field containing label value
    POSITIVE_LABEL : int
        positive sentiment label
    NEGATIVE_LABEL : int
        negative sentiment label
    """

    NAME = "imdb"
    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    DATASET_DIR = os.path.join("imdb", "aclImdb")
    ARCHIVE_TYPE = "tar"
    TRAIN_DIR = "train"
    TEST_DIR = "test"
    POSITIVE_LABEL_DIR = "pos"
    NEGATIVE_LABEL_DIR = "neg"

    TEXT_FIELD_NAME = "text"
    LABEL_FIELD_NAME = "label"

    POSITIVE_LABEL = "positive"
    NEGATIVE_LABEL = "negative"

    def __init__(self, dir_path, fields):
        """
        Dataset constructor. User should use static method get_dataset_splits
        rather than using directly constructor.

        Parameters
        ----------
        dir_path : str
            path to the directory containing datasets

        fields : dict(str, Field)
            dictionary that maps field name to the field
        """
        LargeResource(
            **{
                LargeResource.RESOURCE_NAME: IMDB.NAME,
                LargeResource.ARCHIVE: IMDB.ARCHIVE_TYPE,
                LargeResource.URI: IMDB.URL,
            }
        )
        examples = self._create_examples(dir_path=dir_path, fields=fields)
        super(IMDB, self).__init__(**{"examples": examples, "fields": fields})

    @staticmethod
    def _create_examples(dir_path, fields):
        """
        Method creates examples for imdb dataset. Examples are arranged in two
        folders, one for examples with positive sentiment and other with
        negative sentiment. One file in each folder represents one example.

        Parameters
        ----------
        dir_path : str
            directory where files with examples are positioned
        fields : dict(str, Field)
            dictionary mapping field names to fields

        Returns
        -------
        examples : list(Example)
            list of examples from given dir_path
        """
        dir_pos_path = os.path.join(dir_path, IMDB.POSITIVE_LABEL_DIR)
        dir_neg_path = os.path.join(dir_path, IMDB.NEGATIVE_LABEL_DIR)
        examples = []
        examples.extend(
            IMDB._create_labeled_examples(dir_pos_path, IMDB.POSITIVE_LABEL, fields)
        )
        examples.extend(
            IMDB._create_labeled_examples(dir_neg_path, IMDB.NEGATIVE_LABEL, fields)
        )
        return examples

    @staticmethod
    def _create_labeled_examples(dir_path, label, fields):
        """
        Method creates examples for imdb dataset with given label. Examples are
        positioned in multiple files that are in one folder.

        Parameters
        ----------
        dir_path : str
            file where files with examples are positioned
        label : int
            examples label
        fields : dict(str, Field)
            dictionary mapping field names to fields

        Returns
        -------
        examples : list(Example)
            list of examples from given dir_path
        """
        example_factory = ExampleFactory(fields)
        files_list = [
            f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
        ]
        examples = []
        for file_path in files_list:
            with open(file=os.path.join(dir_path, file_path), encoding="utf8") as fpr:
                data = {IMDB.TEXT_FIELD_NAME: fpr.read(), IMDB.LABEL_FIELD_NAME: label}
                examples.append(example_factory.from_dict(data))
        return examples

    @staticmethod
    def get_dataset_splits(fields=None):
        """
        Method creates train and test dataset for Imdb dataset.

        Parameters
        ----------
        fields : dict(str, Field), optional
            dictionary mapping field name to field, if not given method will
            use ```get_default_fields```. User should use default field names
            defined in class attributes.

        Returns
        -------
        (train_dataset, test_dataset) : (Dataset, Dataset)
            tuple containing train dataset and test dataset
        """
        data_location = os.path.join(LargeResource.BASE_RESOURCE_DIR, IMDB.DATASET_DIR)
        if not fields:
            fields = IMDB.get_default_fields()

        train_dataset = IMDB(
            dir_path=os.path.join(data_location, IMDB.TRAIN_DIR), fields=fields
        )

        test_dataset = IMDB(
            dir_path=os.path.join(data_location, IMDB.TEST_DIR), fields=fields
        )

        return (train_dataset, test_dataset)

    @staticmethod
    def get_default_fields():
        """
        Method returns default Imdb fields: text and label.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """
        text = Field(
            name=IMDB.TEXT_FIELD_NAME,
            numericalizer=Vocab(),
            tokenizer="spacy",
        )
        label = LabelField(name=IMDB.LABEL_FIELD_NAME, numericalizer=Vocab(specials=()))
        return {IMDB.TEXT_FIELD_NAME: text, IMDB.LABEL_FIELD_NAME: label}
