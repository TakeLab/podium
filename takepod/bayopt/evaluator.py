class Evaluator(object):
    """ A simple callable that wraps the evaluation function that is optimized
    with bayesian optimization and stores the evaluation history.

    Attributes
    ----------
    evaluation_function : callable
        The function that gets evaluated when this object is called.

    history : list
        A list of tuples (hyperparameter_dict, result) that represent the
        evaluation history.
    """
    def __init__(self, evaluation_function):
        """
        Creates the callable Evaluator object.

        Parameters
        ----------
        evaluation_function : callable
            The function that will be evaluated each time this object is
            called. The dict passed when calling this object is used as the
            **kwargs to call the evaluation_function, so the
            evaluation_function has to have the correct parameters.
        """

        self.evaluation_function = evaluation_function
        self.history = list()

    def __call__(self, hyperparameter_dict):
        """
        Method takes a dict of hyperparameters and uses it as **kwargs to
        call the evaluation_function and appends the result of the
        evaluation to history as a tuple (hyperparameter_dict, result).

        Parameters
        ----------
        hyperparameter_dict : dict
            The dict of hyperparameter values used as **kwargs to call the
            evaluation_function.

        Returns
        -------
            The result that the evaluation_function returned.
        """

        result = self.evaluation_function(**hyperparameter_dict)
        self.history.append((hyperparameter_dict, result))

        return result
