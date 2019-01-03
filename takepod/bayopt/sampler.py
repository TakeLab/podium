from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import sklearn.gaussian_process as gp
from scipy.optimize import minimize
from scipy.stats import norm


class Sampler(ABC):
    def __init__(self):
        self.history = list()

    @abstractmethod
    def next_sample(self):
        raise NotImplementedError

    def update(self, x, y):
        self.history.append((x, y))


class GaussianProcessSampler(Sampler):
    def __init__(self, hyperparameter_definition, create_model_func=None,
                 acquisition_function=None):
        super().__init__()

        if create_model_func is None:
            create_model_func = _create_gp_model
        self.create_model = create_model_func

        if acquisition_function is None:
            acquisition_function = _expected_improvement
        self.acquisition_function = acquisition_function

        self.hyperparameter_definition = hyperparameter_definition

    def next_sample(self):
        shape = (len(self.history), -1)

        xs, ys = zip(*self.history)
        X, Y = np.array(xs).reshape(shape), np.array(ys).reshape(shape)

        model = self.create_model()
        model.fit(X, Y)

        # create partial acq. func. using model and y_best
        partial_acq_func = partial(self.acquisition_function, model=model,
                                   y_best=Y.max())

        # optimize acq. func.
        best_x, best_y = acquisition_function_maximization(
            partial_acq_func,
            self.hyperparameter_definition.bounds)

        return best_x


def _create_gp_model():
    return gp.GaussianProcessRegressor(
        kernel=gp.kernels.Matern(),
        alpha=1e-5,
        normalize_y=True,
        n_restarts_optimizer=40
    )


def acquisition_function_maximization(acq_func, bounds: np.ndarray):
    n_starting_points = 25
    n_parameters = bounds.shape[0]

    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    sampled_starting_points = np.random.uniform(lower_bounds, upper_bounds,
                                                size=(n_starting_points,
                                                      n_parameters))

    best_result, best_x, result = None, None, None
    # start loop for each starting point
    for starting_x in sampled_starting_points:
        #  call minimize for the starting point with the partial of the
        # acq_fun and its bounds
        result = minimize(
            fun=lambda x: -acq_func(x),  # want to maximize
            x0=starting_x,
            bounds=bounds,
            method='L-BFGS-B'
        )

        #  if the result is better than the current best, set it as best
        if (best_result is None) or (result.fun < best_result):
            best_result = result.fun
            best_x = result.x

    # if some elements of x are lower than their lower bound,
    # set them equal to lower bound
    lower_indices = best_x < lower_bounds
    best_x[lower_indices] = lower_bounds[lower_indices]

    # if some elements of x are larger than their upper bound,
    # set them equal to upper bound
    higher_indices = best_x > upper_bounds
    best_x[higher_indices] = upper_bounds[higher_indices]

    return best_x, best_result


def _expected_improvement(x: np.ndarray, model: gp.GaussianProcessRegressor,
                          y_best: float):
    """
    EI:
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
    """

    x = x.reshape((1, -1))
    mi, sigma = model.predict(x, return_std=True)

    if abs(sigma) < 1e-4:
        return 0.0

    Z = (mi - y_best) / sigma
    PHI = norm.cdf(Z)
    phi = norm.pdf(Z)

    ei = (mi - y_best) * PHI + sigma * phi

    return ei
