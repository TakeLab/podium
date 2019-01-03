class BayesianOptimization(object):
    def __init__(self, hyperparameter_definition, sampler, evaluator,
                 init_hparam_values):

        self.hyperparameter_definition = hyperparameter_definition

        self.hp_initial_values = init_hparam_values

        self.evaluator = evaluator
        self.sampler = sampler

    def perform_bayesian_optimization(self, n_iter=20):
        # evaluating the given initial points and updating history
        for hp_value in self.hp_initial_values:
            result = self.evaluator(hp_value)

            hp_vector = self.hyperparameter_definition.dict_to_real_vector(
                hp_value)

            self.sampler.update(hp_vector, result)

        for i in range(1, n_iter + 1):
            print(f"Iter: {i}")

            # find the best hyperparameter choice to evaluate next,
            # based on the history
            next_sample_real_vector = self.sampler.next_sample()

            # transform real vector to dict
            next_sample = self.hyperparameter_definition.real_vector_to_dict(
                next_sample_real_vector)

            # evaluate the choice of hparams
            result = self.evaluator(next_sample)

            # update the history
            self.sampler.update(next_sample_real_vector, result)

        # get the best found hparams
        best_hparams, best_result = max(self.evaluator.history,
                                        key=lambda el: el[1])

        return best_hparams, best_result

    def get_best_result(self):
        # get the best found hparams
        best_hparams, best_result = max(self.evaluator.history,
                                        key=lambda el: el[1])

        return best_hparams, best_result
