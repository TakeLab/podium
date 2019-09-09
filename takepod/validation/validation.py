from typing import Callable, Optional, Union, List, Tuple
import numpy as np

from takepod.datasets import Dataset, SingleBatchIterator
from takepod.validation import KFold
from takepod.models.experiment import Experiment

import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

_LOGGER = logging.getLogger(__name__)


def kfold_scores(
        experiment: Experiment,
        dataset: Dataset,
        n_splits: int,
        score_fun: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, int, float]],
        shuffle: Optional[bool] = False,
        random_state: int = None) -> List[Union[np.ndarray, int, float]]:
    kfold = KFold(n_splits=n_splits,
                  shuffle=shuffle,
                  random_state=random_state)

    it = SingleBatchIterator()
    results = list()
    for train_split, test_split in kfold.split(dataset):
        experiment.fit(train_split)
        y_pred = experiment.predict(test_split)

        it.set_dataset(test_split)
        _, y_true_batch = next(iter(it))
        y_true = experiment.label_transform_fun(y_true_batch)

        split_score = score_fun(y_true, y_pred)
        results.append(split_score)

    return results


def k_fold_validation(experiment: Experiment,
                      dataset: Dataset,
                      n_splits: int,
                      score_fun: Callable[[np.ndarray, np.ndarray], float],
                      shuffle: Optional[bool] = False,
                      random_state: int = None) -> float:
    # TODO add option to calculate statistical values (variance, p-value...)?
    results = kfold_scores(experiment,
                           dataset,
                           n_splits,
                           score_fun,
                           shuffle,
                           random_state)

    return sum(results) / len(results)


def k_fold_multiclass_metrics(experiment: Experiment,
                              dataset: Dataset,
                              n_splits: int,
                              shuffle: Optional[bool] = False,
                              random_state: int = None,
                              average: str = 'micro') \
        -> Tuple[float, float, float, float]:
    # TODO expand with `binary` from scikit
    if average not in ('micro', 'macro', 'weighted'):
        error_msg = "'average' parameter must be either 'micro', 'macro' or 'weighted'." \
                    " Provided value: '{}'".format(average)
        _LOGGER.error(error_msg)
        raise ValueError(error_msg)

    def scorer(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true,
                                                                   y_pred,
                                                                   average=average)
        return np.array([
            accuracy,
            precision,
            recall,
            f1
        ])

    results = kfold_scores(
        experiment,
        dataset,
        n_splits,
        scorer,
        shuffle,
        random_state
    )

    results_avg = sum(results) / len(results)
    return tuple(results_avg)
