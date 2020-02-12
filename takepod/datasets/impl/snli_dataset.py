"""
Module contains the The Stanford Natural Language Inference (SNLI) Corpus
For more information see: https://nlp.stanford.edu/projects/snli/
"""
import os

from takepod.datasets import Dataset
from takepod.storage import LargeResource, ExampleFactory


class SNLIDataset(Dataset):
    NAME = "snli_1.0"
    URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    DATASET_DIR = "snli_1.0"
    ARCHIVE_TYPE = "zip"
    TRAIN_FILE_NAME = "snli_1.0_train.jsonl"
    TEST_FILE_NAME = "snli_1.0_test.jsonl"
    DEV_FILE_NAME = "snli_1.0_dev.jsonl"

    def __init__(self, file_path, fields):
        print("Downloading file: " + str(file_path))
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
        # TODO: Ovaj print pocistiti u nekom trenutku naravno
        print("Opening file: " + str(file_path))

        #example_factory = ExampleFactory(fields)
        examples = []
        with open(file=file_path, mode='r', encoding='utf8') as in_file:
            for line in in_file.readlines():
                #examples.append(example_factory.from_json(line))
                # TODO: Ovo popraviti kad fieldovi budu gotovi
                print("Parsing example...")
        return examples

    @staticmethod
    def get_train_test_dev_dataset(fields=None):
        data_location = os.path.join(LargeResource.BASE_RESOURCE_DIR,
                                     SNLIDataset.DATASET_DIR)

        if not fields:
            fields = SNLIDataset.get_default_fields()

        train_dataset = SNLIDataset(file_path=os.path.join(
            data_location, SNLIDataset.TRAIN_FILE_NAME), fields=fields)
        test_dataset = SNLIDataset(file_path=os.path.join(
            data_location, SNLIDataset.TEST_FILE_NAME), fields=fields)
        dev_dataset = SNLIDataset(file_path=os.path.join(
            data_location, SNLIDataset.DEV_FILE_NAME), fields=fields)

        # TODO: Finalize nad svima ili samo nad train? U pauzi je nad train
        train_dataset.finalize_fields()
        test_dataset.finalize_fields()
        dev_dataset.finalize_fields()

        return (train_dataset, test_dataset, dev_dataset)

    @staticmethod
    def get_default_fields():
        # TODO: Vidjeti kako definirati tocno ta polja
        return None
