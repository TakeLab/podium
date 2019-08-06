from typing import Callable, Optional, Union
import numpy as np

from takepod.datasets import Dataset, SingleBatchIterator
from takepod.models.experiment import Experiment
from takepod.validation import KFold


def cross_validation_scores(
        dataset: Dataset,
        experiment: Experiment,
        n_splits: int,
        score_fun: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, int, float]],
        shuffle: Optional[bool] = False,
        random_state: int = None):

    kfold = KFold(n_splits=n_splits,
                  shuffle=shuffle,
                  random_state=random_state)

    it = SingleBatchIterator()
    resutls = list()
    for train_split, test_split in kfold.split(dataset):
        experiment.fit(train_split)
        y_pred = experiment.predict(test_split)

        it.set_dataset(test_split)
        _, y_true_batch = next(iter(it))
        y_true = experiment.label_transform_fun(y_true_batch)

        split_score = score_fun(y_pred, y_true)
        resutls.append(split_score)

    return resutls
