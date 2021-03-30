"""
Module contains the The Stanford Natural Language Inference (SNLI) Corpus For
more information about the dataset see: https://nlp.stanford.edu/projects/snli/
or check the README.txt file in the dataset directory.
"""
import os

from podium.datasets import Dataset
from podium.datasets.example_factory import ExampleFactory
from podium.field import Field, LabelField
from podium.storage import LargeResource
from podium.vocab import Vocab


class SNLISimple(Dataset):
    """
    A Simple SNLI Dataset class. This class only uses three fields by default:
    gold_label, sentence1, sentence2.

    Attributes
    ----------
    NAME : str
        Name of the Dataset.
    URL : str
        URL to the SNLI dataset.
    DATASET_DIR : str
        Name of the directory in which the dataset files are stored.
    ARCHIVE_TYPE : str
        Archive type, i.e. compression method used for archiving the
        downloaded dataset file.
    TRAIN_FILE_NAME : str
        Name of the file in which the train dataset is stored.
    TEST_FILE_NAME : str
        Name of the file in which the test dataset is stored.
    DEV_FILE_NAME : str
        Name of the file in which the dev (validation) dataset is stored.

    GOLD_LABEL_FIELD_NAME : str
        Name of the field containing gold label
    SENTENCE1_FIELD_NAME : str
        Name of the field containing sentence1
    SENTENCE2_FIELD_NAME : str
        Name of the field containing sentence2
    """

    NAME = "snli_1.0"
    URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    DATASET_DIR = "snli_1.0"
    ARCHIVE_TYPE = "zip"
    TRAIN_FILE_NAME = "snli_1.0_train.jsonl"
    TEST_FILE_NAME = "snli_1.0_test.jsonl"
    DEV_FILE_NAME = "snli_1.0_dev.jsonl"

    GOLD_LABEL_FIELD_NAME = "gold_label"
    SENTENCE1_FIELD_NAME = "sentence1"
    SENTENCE2_FIELD_NAME = "sentence2"

    def __init__(self, file_path, fields):
        """
        Dataset constructor. This method should not be used directly,
        `get_train_test_dev_dataset` should be used instead.

        Parameters
        ----------
        file_path : str
            Path to the `.jsonl` file containing the dataset.
        fields : dict(str, Field)
            A dictionary that maps field names to Field objects.
        """
        LargeResource(
            **{
                LargeResource.RESOURCE_NAME: SNLISimple.NAME,
                LargeResource.ARCHIVE: SNLISimple.ARCHIVE_TYPE,
                LargeResource.URI: SNLISimple.URL,
            }
        )
        examples = self._create_examples(file_path, fields)
        super(SNLISimple, self).__init__(**{"examples": examples, "fields": fields})

    @staticmethod
    def _create_examples(file_path, fields):
        """
        Method creates Examples for the SNLI Dataset, from a single input
        `.jsonl` file.

        Parameters
        ----------
        file_path : str
            Path to the `.jsonl` file in which examples are stored.
        fields : dict(str, Field)
            A dictionary mapping field names to Field objects.

        Returns
        -------
        examples: list(Example)
            List of examples extracted from the provided `.jsonl` file.
        """

        example_factory = ExampleFactory(fields)
        examples = []

        with open(file=file_path, encoding="utf8") as in_file:
            for line in in_file:
                examples.append(example_factory.from_json(line))
        return examples

    @staticmethod
    def get_train_test_dev_dataset(fields=None):
        """
        Method creates train, test and dev (validation) Datasets for the SNLI
        dataset. If the `snli_1.0` directory is not present in the
        current/working directory, it will be downloaded automatically.

        Parameters
        ----------
        fields : dict(str, Field), optional
            A dictionary that maps field names to Field objects.
            If not supplied, ```get_default_fields``` is used.

        Returns
        -------
        (train_dataset, test_dataset, dev_dataset) : (Dataset, Dataset, Dataset)
            A tuple containing train, test and dev Datasets respectively.
        """
        data_location = os.path.join(
            LargeResource.BASE_RESOURCE_DIR, SNLISimple.DATASET_DIR
        )

        if not fields:
            fields = SNLISimple.get_default_fields()

        train_dataset = SNLISimple(
            file_path=os.path.join(data_location, SNLISimple.TRAIN_FILE_NAME),
            fields=fields,
        )
        test_dataset = SNLISimple(
            file_path=os.path.join(data_location, SNLISimple.TEST_FILE_NAME),
            fields=fields,
        )
        dev_dataset = SNLISimple(
            file_path=os.path.join(data_location, SNLISimple.DEV_FILE_NAME), fields=fields
        )

        return train_dataset, test_dataset, dev_dataset

    @staticmethod
    def get_default_fields():
        """
        Method returns the three main SNLI fields in the following order:
        gold_label, sentence1, sentence2.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field names to respective Fields.
        """

        gold_label = LabelField(
            name=SNLISimple.GOLD_LABEL_FIELD_NAME, numericalizer=Vocab(specials=())
        )
        sentence_vocab = Vocab()
        sentence1 = Field(
            name=SNLISimple.SENTENCE1_FIELD_NAME,
            numericalizer=sentence_vocab,
            tokenizer="split",
            keep_raw=False,
        )
        sentence2 = Field(
            name=SNLISimple.SENTENCE2_FIELD_NAME,
            numericalizer=sentence_vocab,
            tokenizer="split",
            keep_raw=False,
        )
        fields = {
            SNLISimple.GOLD_LABEL_FIELD_NAME: gold_label,
            SNLISimple.SENTENCE1_FIELD_NAME: sentence1,
            SNLISimple.SENTENCE2_FIELD_NAME: sentence2,
        }
        return fields


class _TreeFactory:
    """
    Used for creating trees from strings.

    This class is necessary to make the dataset pickleable.
    """

    def __call__(self, text):

        try:
            from nltk import Tree
        except ImportError:
            print(
                "Problem occurred while trying to import nltk. "
                "If the library is not installed visit "
                "https://www.nltk.org/ for more details."
            )
            raise

        if text[0] != "(":
            text = "(" + text + ")"
        return Tree.fromstring(text)


# TODO: Write tests for SNLI
class SNLI(SNLISimple):
    """
    A SNLI dataset class. Unlike `SNLISimple`, this class includes all the
    fields included in the SNLI dataset by default.

    Attributes
    ----------
    NAME : str
        Name of the Dataset.
    URL : str
        URL to the SNLI dataset.
    DATASET_DIR : str
        Name of the directory in which the dataset files are stored.
    ARCHIVE_TYPE : str
        Archive type, i.e. compression method used for archiving the
        downloaded dataset file.
    TRAIN_FILE_NAME : str
        Name of the file in which the train dataset is stored.
    TEST_FILE_NAME : str
        Name of the file in which the test dataset is stored.
    DEV_FILE_NAME : str
        Name of the file in which the dev (validation) dataset is stored.

    ANNOTATOR_LABELS_FIELD_NAME : str
        Name of the field containing annotator labels
    CAPTION_ID_FIELD_NAME : str
        Name of the field containing caption ID
    GOLD_LABEL_FIELD_NAME : str
        Name of the field containing gold label
    PAIR_ID_FIELD_NAME : str
        Name of the field containing pair ID
    SENTENCE1_FIELD_NAME : str
        Name of the field containing sentence1
    SENTENCE1_PARSE_FIELD_NAME : str
        Name of the field containing sentence1 parse
    SENTENCE1_BINARY_PARSE_FIELD_NAME : str
        Name of the field containing sentence1 binary parse
    SENTENCE2_FIELD_NAME : str
        Name of the field containing sentence2
    SENTENCE2_PARSE_FIELD_NAME : str
        Name of the field containing sentence2 parse
    SENTENCE2_BINARY_PARSE_FIELD_NAME : str
        Name of the field containing sentence2 binary parse
    """

    ANNOTATOR_LABELS_FIELD_NAME = "annotator_labels"
    CAPTION_ID_FIELD_NAME = "captionID"
    GOLD_LABEL_FIELD_NAME = "gold_label"
    PAIR_ID_FIELD_NAME = "pairID"
    SENTENCE1_FIELD_NAME = "sentence1"
    SENTENCE1_PARSE_FIELD_NAME = "sentence1_parse"
    SENTENCE1_BINARY_PARSE_FIELD_NAME = "sentence1_binary_parse"
    SENTENCE2_FIELD_NAME = "sentence2"
    SENTENCE2_PARSE_FIELD_NAME = "sentence2_parse"
    SENTENCE2_BINARY_PARSE_FIELD_NAME = "sentence2_binary_parse"

    @staticmethod
    def get_train_test_dev_dataset(fields=None):
        if fields is None:
            fields = SNLI.get_default_fields()
        return SNLISimple.get_train_test_dev_dataset(fields)

    @staticmethod
    def get_default_fields():
        """
        Method returns all SNLI fields in the following order: annotator_labels,
        captionID, gold_label, pairID, sentence1, sentence1_parse,
        sentence1_binary_parse, sentence2, sentence2_parse,
        sentence2_binary_parse.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field names to respective Fields.

        Notes
        -----
        This dataset includes both parses for every sentence,
        """

        tree_factory = _TreeFactory()
        fields = SNLISimple.get_default_fields()
        annotator_labels = Field(
            name=SNLI.ANNOTATOR_LABELS_FIELD_NAME,
            tokenizer=None,
            numericalizer=Vocab(specials=()),
        )
        captionID = Field(
            name=SNLI.CAPTION_ID_FIELD_NAME, tokenizer=lambda x: x, keep_raw=False
        )
        pairID = Field(
            name=SNLI.PAIR_ID_FIELD_NAME, tokenizer=lambda x: x, keep_raw=False
        )
        sentence1_parse = Field(
            name=SNLI.SENTENCE1_PARSE_FIELD_NAME,
            tokenizer=tree_factory,
            keep_raw=False,
        )
        sentence1_binary_parse = Field(
            name=SNLI.SENTENCE1_BINARY_PARSE_FIELD_NAME,
            tokenizer=tree_factory,
            keep_raw=False,
        )
        sentence2_parse = Field(
            name=SNLI.SENTENCE2_PARSE_FIELD_NAME,
            tokenizer=tree_factory,
            keep_raw=False,
        )
        sentence2_binary_parse = Field(
            name=SNLI.SENTENCE2_BINARY_PARSE_FIELD_NAME,
            tokenizer=tree_factory,
            keep_raw=False,
        )

        fields.update(
            {
                SNLI.ANNOTATOR_LABELS_FIELD_NAME: annotator_labels,
                SNLI.CAPTION_ID_FIELD_NAME: captionID,
                SNLI.PAIR_ID_FIELD_NAME: pairID,
                SNLI.SENTENCE1_PARSE_FIELD_NAME: sentence1_parse,
                SNLI.SENTENCE1_BINARY_PARSE_FIELD_NAME: sentence1_binary_parse,
                SNLI.SENTENCE2_PARSE_FIELD_NAME: sentence2_parse,
                SNLI.SENTENCE2_BINARY_PARSE_FIELD_NAME: sentence2_binary_parse,
            }
        )
        return fields
