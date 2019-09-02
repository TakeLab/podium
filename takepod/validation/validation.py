from typing import Callable, Optional, Union, List, Tuple
from functools import partial
import numpy as np

from takepod.datasets import Dataset, SingleBatchIterator
from takepod.validation import KFold
from takepod.models.experiment import Experiment

import logging
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score

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

    accuracy_scorer = accuracy_score
    precision_scorer = partial(precision_score, average=average)
    recall_scorer = partial(recall_score, average=average)
    f1_scorer = partial(f1_score, average=average)

    def scorer(y_true, y_pred):
        return np.array([
            accuracy_scorer(y_true, y_pred),
            precision_scorer(y_true, y_pred),
            recall_scorer(y_true, y_pred),
            f1_scorer(y_true, y_pred)
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
