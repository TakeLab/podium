from torchtext.datasets.imdb import IMDB
import glob
import os


class Imdb(IMDB):

    def __init__(self, path="downloaded_datasets", **kwargs):
        self._data_dir = path
        text_field = "text"
        label_field = "sentiment"
        super(Imdb, self).__init__(
            self._data_dir, text_field, label_field, **kwargs
        )

    def download_and_extract(self):
        # TODO this is very slow even when it
        # doesn't download, find a way to speed this up
        self.download(root="downloaded_datasets",
                      check="downloaded_datasets/imdb")

    def _read_text_file(self, path):
        """
        Read and return all the contents of the text-file with the given path.
        It is returned as a single string where all lines are concatenated.
        """
        with open(path, 'rt') as file:
            # Read a list of strings.
            lines = file.readlines()
            # Concatenate to a single string.
            text = " ".join(lines)
        return text

    def load_data(self, train=True):
        """
        Load all the data from the IMDB Review data-set for sentiment analysis.

        Parameters
        ----------
        train: bool
            Determines whether to load the training-set (True)
            or the test-set (False).

        Returns
        -------
        x : list of str
            List of text-strings of reviews
        y : list of str
            list of the review corresponding sentiments
            with Positive and Negative values
        """

        # Part of the path-name for either training or test-set.
        train_test_path = "train" if train else "test"

        # Base-directory where the extracted data is located.
        print(self._data_dir)
        dir_base = os.path.join(self._data_dir,
                                "imdb", "aclImdb", train_test_path)
        print(dir_base)

        # Filename-patterns for the data-files.
        path_pattern_pos = os.path.join(dir_base, "pos", "*.txt")
        path_pattern_neg = os.path.join(dir_base, "neg", "*.txt")

        # Get lists of all the file-paths for the data.
        paths_pos = glob.glob(path_pattern_pos)
        paths_neg = glob.glob(path_pattern_neg)

        # Read all the text-files.
        data_pos = [self._read_text_file(path) for path in paths_pos]
        data_neg = [self._read_text_file(path) for path in paths_neg]

        # Concatenate the positive and negative data.
        x = data_pos + data_neg

        # Create a list of the sentiments for the text-data.
        y = ['Positive'] * len(data_pos) + ['Negative'] * len(data_neg)

        return x, y
