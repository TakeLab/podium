from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

from podium.datasets import Dataset
from podium.experimental.models.experiment import Experiment


class _KFold(KFold):
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


def kfold_scores(
    experiment: Experiment,
    dataset: Dataset,
    n_splits: int,
    score_fun: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, int, float]],
    shuffle: Optional[bool] = False,
    random_state: int = None,
) -> List[Union[np.ndarray, int, float]]:
    """
    Calculates a score for each train/test fold. The score for a fold is
    calculated by first fitting the experiment to the train split and then using
    the test split to calculate predictions and evaluate the score. This is
    repeated for every fold.

    Parameters
    ----------
    experiment : Experiment
        Experiment defining the training and prediction procedure to be evaluated.

    dataset : Dataset
        Dataset to be used for experiment evaluation.

    n_splits : int
        Number of folds.

    score_fun : Callable (y_true, y_predicted) -> score
        Callable used to evaluate the score for a fold. This callable should take
        two numpy array arguments: y_true and y_predicted. y_true is the ground
        truth while y_predicted are the model's predictions. This callable should
        return a score that can be either a numpy array, a int or a float.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.

    Returns
    -------
        a List of scores provided by score_fun for every fold.
    """
    kfold = _KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    results = []
    for train_split, test_split in kfold.split(dataset):
        experiment.fit(train_split)
        y_pred = experiment.predict(test_split)

        _, y_true_batch = test_split.batch()
        y_true = experiment.label_transform_fn(y_true_batch)

        split_score = score_fun(y_true, y_pred)
        results.append(split_score)

    return results


def k_fold_validation(
    experiment: Experiment,
    dataset: Dataset,
    n_splits: int,
    score_fun: Callable[[np.ndarray, np.ndarray], float],
    shuffle: Optional[bool] = False,
    random_state: int = None,
) -> Union[np.ndarray, int, float]:
    # TODO add option to calculate statistical values (variance, p-value...)?
    """
    Convenience function for kfold_scores. Calculates scores for every fold and
    returns the mean of all scores.

    Parameters
    ----------
    experiment : Experiment
        Experiment defining the training and prediction procedure to be evaluated.

    dataset : Dataset
        Dataset to be used for experiment evaluation.

    n_splits : int
        Number of folds.

    score_fun : Callable (y_true, y_predicted) -> score
        Callable used to evaluate the score for a fold. This callable should take
        two numpy array arguments: y_true and y_predicted. y_true is the ground
        truth while y_predicted are the model's predictions. This callable should
        return a score that can be either a numpy array, a int or a float.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.

    Returns
    -------
        The mean of all scores for every fold.
    """
    results = kfold_scores(
        experiment, dataset, n_splits, score_fun, shuffle, random_state
    )

    return sum(results) / len(results)


def k_fold_classification_metrics(
    experiment: Experiment,
    dataset: Dataset,
    n_splits: int,
    average: str = "micro",
    beta: float = 1.0,
    labels: List[int] = None,
    pos_label: int = 1,
    shuffle: Optional[bool] = False,
    random_state: int = None,
) -> Tuple[float, float, float, float]:
    """Calculates the most often used classification metrics : accuracy, precision,
    recall and the F1 score. All scores are calculated for every fold and the mean
    of every score over all folds is returned.

    Parameters
    ----------
    experiment : Experiment
        Experiment defining the training and prediction procedure to be evaluated.

    dataset : Dataset
        Dataset to be used for experiment evaluation.

    n_splits : int
        Number of folds.

    average : str, Optional
        Determines the type of averaging performed.

        The supported averaging methods are:

        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.

        'macro':
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.

        'weighted':
            Calculate metrics for each label, and find their average weighted by support
            (the number of true instances for each label). This alters ‘macro’ to account
            for label imbalance; it can result in an F-score that is not between precision
            and recall.

        `binary`:
            Only report results for the class specified by pos_label.
            This is applicable only if targets (i.e. results of predict) are binary.

        None:
            The scores for each class are returned.

    beta: float
        The strength of recall versus precision in the F-score.

    labels : List, optional
        The set of labels to include when average != 'binary', and their order if average
        is None. Labels present in the data can be excluded, for example to calculate a
        multiclass average ignoring a majority negative class, while labels not present in
        the data will result in 0 components in a macro average. For multilabel targets,
        labels are column indices.

    pos_label: int
        The class to report if average='binary' and the data is binary. If the data are
        multiclass or multilabel, this will be ignored; setting labels=[pos_label] and
        average != 'binary' will report scores for that label only.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.

    Returns
    -------
    tuple(float, float, float, float)
        A tuple containing four classification metrics: accuracy, precision, recall, f1
        Each score returned is a mean of that score over all folds.

    Raises
    ------
    ValueError
        If `average` is not one of: `micro`, `macro`, `weighted`, `binary`
    """
    if average not in ("micro", "macro", "weighted", "binary"):
        raise ValueError(
            "`average` parameter must be either `micro`, `macro`, `weighted` "
            f"or `binary`. Provided value: '{average}'"
        )

    def scorer(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, pos_label=pos_label, average=average, beta=beta
        )
        return np.array([accuracy, precision, recall, f1])

    results = kfold_scores(experiment, dataset, n_splits, scorer, shuffle, random_state)

    results_avg = sum(results) / len(results)
    return tuple(results_avg)
