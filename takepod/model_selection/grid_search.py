from typing import Callable, Iterable, Tuple, Dict, Any, Union

import numpy as np
from sklearn.model_selection import ParameterGrid
import logging
from tqdm import tqdm

from takepod.models import Experiment
from takepod.datasets import Dataset
from takepod.validation import k_fold_validation

_LOGGER = logging.getLogger(__name__)


def grid_search(
        experiment: Experiment,
        dataset: Dataset,
        score_fun: Callable[[np.ndarray, np.ndarray], float],
        model_param_grid: Union[Dict[str, Iterable], Iterable[Dict[str, Iterable]]],
        trainer_param_grid: Union[Dict[str, Iterable], Iterable[Dict[str, Iterable]]],
        n_splits: int = 5,
        greater_is_better: bool = True,
        print_progress: bool = True) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    # TODO: Add dataset shuffling to cross-validation

    best_model_params = None
    best_train_params = None
    best_score = None

    try:
        trainer_grid_iter = ParameterGrid(trainer_param_grid)
        model_grid_iter = ParameterGrid(model_param_grid)

    except TypeError as err:
        _LOGGER.error(str(err))
        raise err

    if print_progress:
        pbar = tqdm(desc="Grid search",
                    total=len(model_grid_iter) * len(trainer_grid_iter))

    for trainer_params in trainer_grid_iter:
        experiment.set_default_trainer_args(**trainer_params)

        for model_params in model_grid_iter:
            experiment.set_default_model_args(**model_params)
            score = k_fold_validation(experiment,
                                      dataset,
                                      n_splits=n_splits,
                                      score_fun=score_fun)
            if best_score is None\
                    or greater_is_better and score > best_score\
                    or not greater_is_better and score < best_score:
                best_score = score
                best_model_params = model_params
                best_train_params = trainer_params

            if print_progress:
                pbar.update()
                pbar.set_postfix({'Best score': best_score})

    if print_progress:
        pbar.close()

    return best_score, best_model_params, best_train_params
