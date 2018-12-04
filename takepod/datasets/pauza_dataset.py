"""Module contains PauzaHR datasets."""
import os
import tempfile
import zipfile
from takepod.storage import dataset
from takepod.storage.example import Example
from takepod.storage.field import Field
from takepod.storage.vocab import Vocab
from takepod.storage.downloader import SimpleHttpDownloader


class PauzaHRDataset(dataset.Dataset):
    """Simple PauzaHR dataset class which uses original reviews.

    Attributes
    ----------
    URL : str
        url to the PauzaHR dataset
    NAME : str
        dataset name
    TRAIN_DIR : str
        name of the training directory
    TEST_DIR : str
        name of the directory containing test examples
    """

    URL = "http://takelab.fer.hr/data/cropinion/CropinionDataset.zip"
    NAME = "croopinion"
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
        unpacked_fields = dataset.unpack_fields(fields)
        examples = self._create_examples(dir_path, fields)
        super(PauzaHRDataset, self).__init__(examples, unpacked_fields)

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
        examples = []
        for file_path in files_list:
            with open(file=os.path.join(dir_path, file_path),
                      mode='r', encoding='utf8') as fpr:
                examples.append(Example.fromxmlstr(data=fpr.read(),
                                                   fields=fields))
        return examples

    @staticmethod
    def get_train_test_dataset(dir_path):
        """Method creates train and test dataset for PauzaHR dataset.

        Parameters
        ----------
        dir_path : str
            datasets directory

        Returns
        -------
        (train_dataset, test_dataset, fields) : (Dataset, Dataset, dict)
            tuple containing train dataset, test dataset and their fields
        """
        data_location = os.path.join(dir_path,
                                     "croopinion", "CropinionDataset",
                                     "reviews_original")
        if not os.path.exists(path=data_location):
            PauzaHRDataset._download_and_extract(dir_path=dir_path)

        rating = Field(name="Rating", vocab=Vocab(specials=()),
                       sequential=False, store_raw=True)
        source = Field(name="Source", vocab=Vocab(specials=()),
                       sequential=False, store_raw=True)
        text = Field(name="Text", vocab=Vocab(), tokenizer='split',
                     language="hr", sequential=True, store_raw=False)

        fields = {"Text": text, "Rating": rating, "Source": source}

        train_dataset = PauzaHRDataset(dir_path=os.path.join(
            data_location, PauzaHRDataset.TRAIN_DIR), fields=fields)
        test_dataset = PauzaHRDataset(dir_path=os.path.join(
            data_location, PauzaHRDataset.TEST_DIR), fields=fields)
        return (train_dataset, test_dataset, fields)

    @staticmethod
    def _download_and_extract(dir_path):
        """Method downloades and extracts PauzaHR dataset.

        Parameters
        ----------
        dir_path : str
            path to datasets directory
        """
        os.makedirs(name=dir_path)
        download_dir = tempfile.mkdtemp()
        SimpleHttpDownloader.download(uri=PauzaHRDataset.URL,
                                      path=download_dir, overwrite=False)
        zip_ref = zipfile.ZipFile(download_dir, 'r')
        zip_ref.extractall(dir_path)
        zip_ref.close()
