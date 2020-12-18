import numpy as np
from sklearn.model_selection import KFold as KFold_


class KFold(KFold_):
    """
    Adapter class for the scikit-learn KFold class.

    Works with podium datasets directly.
    """

    def split(self, dataset):
        """
        Splits the dataset into multiple train and test folds often used in
        model validation.

        Parameters
        ----------
        dataset : dataset
            The dataset to be split into folds.

        Yields
        -------
        train_set, test_set
            Yields the train and test datasets for every fold.
        """
        indices = np.arange(len(dataset))
        for train_indices, test_indices in super().split(indices):
            yield dataset[train_indices], dataset[test_indices]
