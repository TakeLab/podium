from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from podium.datasets import Dataset
from podium.experimental.models import Experiment
from podium.experimental.validation import k_fold_validation


def grid_search(
    experiment: Experiment,
    dataset: Dataset,
    score_fun: Callable[[np.ndarray, np.ndarray], float],
    model_param_grid: Union[Dict[str, Iterable], Iterable[Dict[str, Iterable]]],
    trainer_param_grid: Union[Dict[str, Iterable], Iterable[Dict[str, Iterable]]],
    n_splits: int = 5,
    greater_is_better: bool = True,
    print_progress: bool = True,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Method used to find the best combination of training and model hyperparameters
    out of the given hyperparameters. This method uses simple grid search to evaluate
    all possible combinations of the given hyperparameters. Based on sklearn's
    :class:`sklearn.model_selection.GridSearchCV`. Each hyperparameter combination is
    scored by first training the model and then evaluating that model using k-fold cross
    validation. The final score for a set of hyperparameters is the mean of scores across
    all folds.


    Parameters
    ----------
    experiment : Experiment
        Experiment defining the training and prediction procedure to be optimised.

    dataset : Dataset
        Dataset to be used in the hyperparameter search.

    score_fun : callable
        Function used to score a hyperparameter set.

    model_param_grid : Dict or Iterable of Dicts
        The model parameter grid. Combinations taken from this grid are passed to the
        model's __init__ function. Dictionary with parameters names (string) as keys and
        lists of parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are explored.
        This enables searching over any sequence of parameter settings.

    trainer_param_grid : Dict or Iterable of Dicts
        The trainer parameter grid. Combinations taken from this grid are passed to the
        trainers's train function. Dictionary with parameters names (string) as keys and
        lists of parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are explored.
        This enables searching over any sequence of parameter settings.

    n_splits : int
        Number of folds to be used in cross-validation.

    greater_is_better : bool
        Whether score_func is a score function (default), meaning high is good, or a loss
        function, meaning low is good.

    print_progress : bool
        Whether to print progress. Progress is printed to sys.stderr.


    See Also
    --------
        :class:`GridSearchCV`
    """
    # TODO: Add dataset shuffling to cross-validation

    best_model_params = None
    best_train_params = None
    best_score = None

    trainer_grid_iter = ParameterGrid(trainer_param_grid)
    model_grid_iter = ParameterGrid(model_param_grid)

    if print_progress:
        pbar = tqdm(
            desc="Grid search", total=len(model_grid_iter) * len(trainer_grid_iter)
        )

    for trainer_params in trainer_grid_iter:
        experiment.set_default_trainer_args(**trainer_params)

        for model_params in model_grid_iter:
            experiment.set_default_model_args(**model_params)
            score = k_fold_validation(
                experiment, dataset, n_splits=n_splits, score_fun=score_fun
            )
            if (
                best_score is None
                or greater_is_better
                and score > best_score
                or not greater_is_better
                and score < best_score
            ):
                best_score = score
                best_model_params = model_params
                best_train_params = trainer_params

                if print_progress:
                    pbar.set_postfix({"Best score": best_score})

            if print_progress:
                pbar.update()

    if print_progress:
        pbar.close()

    experiment.set_default_trainer_args()
    experiment.set_default_model_args()
    return best_score, best_model_params, best_train_params
