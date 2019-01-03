"""
A small demo of the bayesian optimization procedure. Can be deleted later.
"""

from collections import namedtuple

from takepod.bayopt.bayesian_optimization import BayesianOptimization
from takepod.bayopt.evaluator import Evaluator
from takepod.bayopt.hyperparameter_definition import HyperparameterDefinition
from takepod.bayopt.sampler import GaussianProcessSampler


class MockTrainer:
    def __init__(self):
        self.fixed_hparam_values = {
            "x": 1.5,
        }

        self.active_hparam_init_values = {
            "y": 1.2,
            "z": 1,
            "w": "yes"
        }

        HParamInfo = namedtuple("HParamInfo", ["type", "domain"])

        self.active_hparam_info = {
            "y": HParamInfo("real", (-1.0, 2.0)),
            "z": HParamInfo("integer", (0, 3)),
            "w": HParamInfo("categorical", ("no", "yes", "maybe")),
        }

    def get_hyperparameters(self):
        return self.active_hparam_info, self.active_hparam_init_values

    def train(self, **active_hyperparameter_values):
        all_hyperparameter_values = dict()

        all_hyperparameter_values.update(active_hyperparameter_values)
        all_hyperparameter_values.update(self.fixed_hparam_values)

        return self._some_function(**all_hyperparameter_values)

    def _some_function(self, x, y, z, w):
        result = x + (y ** 2) - (z ** 3)

        if w == "yes":
            result += 1
        elif w == "no":
            result -= 1

        return result


if __name__ == '__main__':
    # 1. create model and / or trainer
    trainer = MockTrainer()

    # 2. extract from trainer and/or model:
    #   - definitions of active hyperparameters: type and domain
    #   - values of active hyperparameters
    hparam_info, hparam_init_values = trainer.get_hyperparameters()

    # 3. create the HyperparameterDefinition from the info
    hparam_def = HyperparameterDefinition(hparam_info)

    # 3. create Evaluator
    #    - initalize with an eval. func (wraps the whole evaluation)
    evaluator = Evaluator(trainer.train)

    # 4. create GaussianProcessSampler
    #    - can be initialized with a custom acq. func. or GP model
    sampler = GaussianProcessSampler(hparam_def)

    # 5. create a BayesianOptimization object
    #    - give it initial values of the hyperparameter to evaluate before
    #    starting the bayesian optimization loop
    bayopt = BayesianOptimization(hparam_def, sampler, evaluator,
                                  init_hparam_values=[
                                      hparam_init_values])

    # 6. perform bayesian optimization and get the best result
    bayopt.perform_bayesian_optimization(n_iter=20)
    best_hparam_values, best_result = bayopt.get_best_result()
