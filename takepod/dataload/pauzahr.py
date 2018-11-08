"""Simple PauzaHR dataset module."""
import os
import glob
import zipfile
import urllib.request
import xml.etree.ElementTree as ET


class PauzaHR():
    """Simple PauzaHR dataset class

    Attributes
    ----------
    URL : str
        url to PauzaHR dataset storage

    """
    URL = "http://takelab.fer.hr/data/cropinion/CropinionDataset.zip"

    def __init__(self, path="downloaded_datasets/"):
        """IMDB dataset constructor.

        Parameters
        ----------
        path : str
            path to folder where the dataset should be downloaded or loaded
            from if it is already downloaded

        """
        self._data_dir = path

    def download_and_extract(self):
        """Method downloadeds and unzips dataset
           archive if it doesn't exist."""
        download_location = os.path.join(self._data_dir, "croopinion")
        if not os.path.isdir(download_location):
            os.makedirs(download_location)
        else:
            print("Already downloaded; try loading the dataset...")
            return

        print("Downloading the Cropinion dataset")
        zip_location = os.path.join(download_location, 'CropinionDataset.zip')
        urllib.request.urlretrieve(self.URL, zip_location)

        print("Unzipping the Croopinion dataset")
        zip_ref = zipfile.ZipFile(zip_location, 'r')
        zip_ref.extractall(download_location)
        zip_ref.close()

    def load_data(self, train=True):
        """
        Load all the data from the Cropinion Review data-set
        for sentiment analysis.
        More info on http://takelab.fer.hr/data/cropinion/

        Parameters
        ----------
        train: bool
            Determines whether to load the training-set (True)
            or the test-set (False).

        Returns
        -------
        data : list of 2-tuple of (str, str)
            List of 2-tuple strings of pairs of reviews and
            corresponding source
        labels : list of float
            list of the review corresponding ratings [0.0, 6.0]
        """
        datalocation = os.path.join(
            self._data_dir, "croopinion",
            "CropinionDataset", "reviews_original"
        )
        if train:
            datalocation = os.path.join(datalocation, "Train")
        # only train vs. test split
        elif not train:
            datalocation = os.path.join(datalocation, "Test")

        data = []
        labels = []
        for xmlfile in glob.glob(datalocation + "/*.xml"):
            root = ET.parse(xmlfile)
            review_text = root.find('Text').text
            source = root.find('Source').text
            rating = root.find('Rating').text
            print(source, rating, review_text)
            data.append((review_text, source))
            labels.append(float(rating))
        return data, labels
