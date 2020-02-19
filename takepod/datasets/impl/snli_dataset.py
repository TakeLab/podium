"""
Module contains the The Stanford Natural Language Inference (SNLI) Corpus
For more information about the dataset see: https://nlp.stanford.edu/projects/snli/
"""
import os

from takepod.datasets import Dataset
from takepod.storage import LargeResource, ExampleFactory, Field, Vocab, TokenizedField


class SNLIDataset(Dataset):
    """
    A SNLI Dataset class.

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
    """
    NAME = "snli_1.0"
    URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    DATASET_DIR = "snli_1.0"
    ARCHIVE_TYPE = "zip"
    TRAIN_FILE_NAME = "snli_1.0_train.jsonl"
    TEST_FILE_NAME = "snli_1.0_test.jsonl"
    DEV_FILE_NAME = "snli_1.0_dev.jsonl"

    def __init__(self, file_path, fields):
        """
        Dataset constructor. This method should not be used
        directly, `get_train_test_dev_dataset` should be used instead.

        Parameters
        ----------
        file_path : str
            Path to the `.jsonl` file containing the dataset.
        fields : dict(str, Field)
            A dictionary that maps field names to Field objects.
        """
        LargeResource(**{
            LargeResource.RESOURCE_NAME: SNLIDataset.NAME,
            LargeResource.ARCHIVE: SNLIDataset.ARCHIVE_TYPE,
            LargeResource.URI: SNLIDataset.URL
        })
        examples = self._create_examples(file_path, fields)
        super(SNLIDataset, self).__init__(
            **{"examples": examples, "fields": fields})

    @staticmethod
    def _create_examples(file_path, fields):
        """
        Method creates Examples for the SNLI Dataset, from a single
        input `.jsonl` file.

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
        with open(file=file_path, mode='r', encoding='utf8') as in_file:
            for line in in_file.readlines():
                examples.append(example_factory.from_json(line))
        return examples

    @staticmethod
    def get_train_test_dev_dataset(fields=None):
        """
        Method creates train, test and dev (validation) Datasets
        for the SNLI dataset. If the `snli_1.0` directory is not
        present in the current/working directory, it will be
        downloaded automatically.

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
        data_location = os.path.join(LargeResource.BASE_RESOURCE_DIR,
                                     SNLIDataset.DATASET_DIR)

        if not fields:
            fields = SNLIDataset.get_default_fields()

        train_dataset = SNLIDataset(file_path=os.path.join(
            data_location, SNLIDataset.TRAIN_FILE_NAME), fields=fields)
        # TODO: maknuti ove sve printove
        print("Done with train...")
        test_dataset = SNLIDataset(file_path=os.path.join(
            data_location, SNLIDataset.TEST_FILE_NAME), fields=fields)
        print("Done with test...")
        dev_dataset = SNLIDataset(file_path=os.path.join(
            data_location, SNLIDataset.DEV_FILE_NAME), fields=fields)
        print("Done with dev...")

        # TODO: Finalize nad svima ili samo nad train? U pauzi je nad train
        train_dataset.finalize_fields()
        test_dataset.finalize_fields()
        dev_dataset.finalize_fields()

        return train_dataset, test_dataset, dev_dataset

    @staticmethod
    def get_default_fields():
        """
        Method returns all SNLI fields in the following order:
        annotator_labels, captionID, gold_label, pairID,
        sentence1, sentence1_parse, sentence1_binary_parse,
        sentence2, sentence2_parse, sentence2_binary_parse

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field names to respective Fields.
        """

        annotator_labels = TokenizedField(name="annotator_labels", vocab=Vocab())
        captionID = Field(name="captionID", vocab=Vocab(),
                          tokenize=False, store_as_raw=True)
        gold_label = Field(name="gold_label", vocab=Vocab(),
                           tokenize=False, store_as_raw=True)
        pairID = Field(name="pairID", vocab=Vocab(),
                       tokenize=False, store_as_raw=True)
        sentence1 = Field(name="sentence1", vocab=Vocab(),
                          tokenizer="split", tokenize=True, store_as_raw=False)
        # TODO: Add tree parsing for sentence_parse and sentence_binary_parse
        sentence1_parse = Field(name="sentence1_parse", vocab=Vocab(),
                                tokenize=False, store_as_raw=True)
        sentence1_binary_parse = Field(name="sentence1_binary_parse",
                                       vocab=Vocab(), tokenize=False,
                                       store_as_raw=True)
        sentence2 = Field(name="sentence2", vocab=Vocab(),
                          tokenizer="split", tokenize=True, store_as_raw=False)
        # TODO: Add tree parsing for sentence_parse and sentence_binary_parse
        sentence2_parse = Field(name="sentence2_parse", vocab=Vocab(),
                                tokenize=False, store_as_raw=True)
        sentence2_binary_parse = Field(name="sentence2_binary_parse",
                                       vocab=Vocab(), tokenize=False,
                                       store_as_raw=True)
        fields = {"annotator_labels": annotator_labels,
                  "captionID": captionID,
                  "gold_label": gold_label,
                  "pairID": pairID,
                  "sentence1": sentence1,
                  "sentence1_parse": sentence1_parse,
                  "sentence1_binary_parse": sentence1_binary_parse,
                  "sentence2": sentence2,
                  "sentence2_parse": sentence2_parse,
                  "sentence2_binary_parse": sentence2_binary_parse}
        return fields
