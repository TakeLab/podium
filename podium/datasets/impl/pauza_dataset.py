"""Module contains PauzaHR datasets."""
import os

from podium.datasets.dataset import Dataset
from podium.storage.field import Field
from podium.storage.vocab import Vocab
from podium.storage.resources.large_resource import LargeResource
from podium.storage.example_factory import ExampleFactory


class PauzaHRDataset(Dataset):
    """Simple PauzaHR dataset class which uses original reviews.

    Attributes
    ----------
    URL : str
        url to the PauzaHR dataset
    NAME : str
        dataset name
    DATASET_DIR : str
        name of the folder in the dataset containing train and test directories
    ARCHIVE_TYPE : str
        string that defines archive type, used for unpacking dataset
    TRAIN_DIR : str
        name of the training directory
    TEST_DIR : str
        name of the directory containing test examples
    """
    NAME = "croopinion"
    URL = "http://takelab.fer.hr/data/cropinion/CropinionDataset.zip"
    DATASET_DIR = os.path.join("croopinion", "CropinionDataset",
                               "reviews_original")
    ARCHIVE_TYPE = "zip"
    TRAIN_DIR = "Train"
    TEST_DIR = "Test"

    def __init__(self, dir_path, fields):
        """Dataset constructor. User should use static method
        get_train_test_dataset rather than using directly constructor.

        Parameters
        ----------
        dir_path : str
            path to the directory containing datasets

        fields : dict(str, Field)
            dictionary that maps field name to the field
        """
        LargeResource(**{
            LargeResource.RESOURCE_NAME: PauzaHRDataset.NAME,
            LargeResource.ARCHIVE: PauzaHRDataset.ARCHIVE_TYPE,
            LargeResource.URI: PauzaHRDataset.URL})
        examples = self._create_examples(dir_path=dir_path, fields=fields)
        super(PauzaHRDataset, self).__init__(
            **{"examples": examples, "fields": fields})

    @staticmethod
    def _create_examples(dir_path, fields):
        """Method creates examples for PauzaHR dataset. Examples are positioned
        in multiple files that are in one folder.

        Parameters
        ----------
        dir_path : str
            file where xml files with examples are positioned
        fields : dict(str, Field)
            dictionary mapping field names to fields

        Returns
        -------
        examples : list(Example)
            list of examples from given dir_path
        """
        files_list = [f for f in os.listdir(dir_path) if os.path.isfile(
            os.path.join(dir_path, f))]
        example_factory = ExampleFactory(fields)
        examples = []
        for file_path in files_list:
            with open(file=os.path.join(dir_path, file_path),
                      encoding='utf8') as fpr:
                examples.append(example_factory.from_xml_str(fpr.read()))
        return examples

    @staticmethod
    def get_train_test_dataset(fields=None):
        """Method creates train and test dataset for PauzaHR dataset.

        Parameters
        ----------
        fields : dict(str, Field), optional
            dictionary mapping field name to field, if not given method will
            use ```get_default_fields```.

        Returns
        -------
        (train_dataset, test_dataset) : (Dataset, Dataset)
            tuple containing train dataset and test dataset
        """
        data_location = os.path.join(LargeResource.BASE_RESOURCE_DIR,
                                     PauzaHRDataset.DATASET_DIR)
        if not fields:
            fields = PauzaHRDataset.get_default_fields()

        train_dataset = PauzaHRDataset(dir_path=os.path.join(
            data_location, PauzaHRDataset.TRAIN_DIR), fields=fields)
        test_dataset = PauzaHRDataset(dir_path=os.path.join(
            data_location, PauzaHRDataset.TEST_DIR), fields=fields)

        train_dataset.finalize_fields()
        return (train_dataset, test_dataset)

    @staticmethod
    def get_default_fields():
        """Method returns default PauzaHR fields: rating, source and text.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """
        rating = Field(name="Rating", numericalizer=Vocab(specials=()),
                       keep_raw=True, is_target=True)
        source = Field(name="Source", numericalizer=Vocab(specials=()),
                       keep_raw=True)
        text = Field(name="Text", numericalizer=Vocab(), tokenizer='split',
                     keep_raw=False)

        fields = {"Text": text, "Rating": rating, "Source": source}
        return fields
