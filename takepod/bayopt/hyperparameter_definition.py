from collections import OrderedDict, namedtuple
import numpy as np


class HyperparameterDefinition(object):
    # TODO: the whole class should be rewritten much more elegantly
    """A class that holds the necessary information regarding the
    hyperparameters (their types and domains), as well as methods to transform
    hyperparameter from one form to another (dict to real vector, etc), that
    are needed in different steps of bayesian optimization.

    Attributes
    ----------
    hyperparameter_dict : OrderedDict
        An ordered dict mapping hyperparameter names to namedtuples containing
        their types and domain.
    bounds : np.ndarray
        An array of shape (N_variables, 2) holding the lower and upper bound
        of each vector component (categorical hyperparameters are represented
        by multiple vector components) in the real-vector representation of
        hyperparameters.
    """
    def __init__(self, hyperparameters: dict):
        # TODO: this may not need to be an OrderedDict, could be a list or sth
        self.hyperparameter_dict = create_ordered_dict(hyperparameters)
        self.bounds = create_bounds_array(self.hyperparameter_dict)

    def dict_to_real_vector(self, hyperparameter_dict: dict):
        """Method that transform the hyperparameter values given as a dict to
        a real-vector representation. The values of real hyperparameters are
        just copied, the values of integer hyperparameters are converted to
        float and the values of hyperparameters are represented with one-hot
        encoding.

        This method is used when we have some initial hyperparameter samples
        that we wish to evaluate to generate a small evaluation history before
        starting the bayesian optimisation loop.

        Parameters
        ----------
        hyperparameter_dict : dict
            A dict mapping the name of the hyperparameter to its value.

        Returns
        -------
        np.ndarray
            A real vector representation of the hyperparameter dict.
        """
        real_vector = np.zeros(self.bounds.shape[0])

        i = 0
        hparam_dict_items = self.hyperparameter_dict.items()
        for name, (_, hp_type, bounds, cat2idx, _) in hparam_dict_items:
            hp_value = hyperparameter_dict[name]

            if hp_type in {"real", "integer"}:
                hp_value = float(hp_value)
                real_vector[i] = hp_value

                i += 1
            else:
                n_categories = len(cat2idx.keys())
                real_vector[i + cat2idx[hp_value]] = 1.0

                i += n_categories

        return real_vector

    def real_vector_to_dict(self, real_vector: np.ndarray):
        """Method that transform the hyperparameter values given as a real
        vector to a dict representation. What is meant by this is that
        the values of real hyperparameters are copied, the values of integer
        hyperparameters are rounded to the nearest integer (within the bounds)
        and the values of variables corresponding to the categorical
        hyperparameters are transformed to a discrete value from the domain
        whose real-vector component has the maxmimum value.

        This method is used after the Sampler samples a new hyperparameter
        choice and the evaluation function has to be evaluated with it. The
        choice is a real-vector (Sampler works only with real vectors) and
        what the evaluation function needs is a dict with real, integer and
        categorical values of hyperparameters.

        Parameters
        ----------
        real_vector : np.ndarray
            A vector of real numbers that is to be transformed into a dict.

        Returns
        -------
        np.ndarray
            A dict representation of the real vector.
        """

        values_dict = {}

        i = 0
        hparam_dict_items = self.hyperparameter_dict.items()
        for name, (_, hp_type, bounds, _, idx2cat) in hparam_dict_items:
            if hp_type in {"real", "integer"}:
                real_value = real_vector[i]

                value = real_value
                if hp_type == "integer":
                    value = int(round(value))

                values_dict[name] = value

                i += 1
            else:
                n_categories = len(idx2cat.keys())
                subarray = real_vector[i:i + n_categories]
                max_idx = np.argmax(subarray)
                values_dict[name] = idx2cat[max_idx]

                i += n_categories

        return values_dict

    def real_vector_to_mixed_vector(self, real_vector):
        """Method that transform the hyperparameter values given as a real
        vector to a mixed-vector representation. What is meant by this is that
        the values of real hyperparameters are copied, the values of integer
        hyperparameters are rounded to the nearest integer (within the bounds)
        and the values of variables corresponding to the categorical
        hyperparameters are set all to 0 except for the maximum one, which is
        set to 1 (one-hot encoding).

        This method should be used inside the Sampler, inside a kernel-wrapper
        to transform the real vectors into mixed vectors before feeding them
        to the kernel, as described by this paper:
        https://arxiv.org/pdf/1805.03463.pdf

        Parameters
        ----------
        real_vector : np.ndarray
            A vector of real numbers that is to be transformed into a
            mixed vector.

        Returns
        -------
        np.ndarray
            A mixed-vector representation of the real vector.
        """

        mixed_vector = np.zeros_like(real_vector)

        i = 0
        hparam_dict_items = self.hyperparameter_dict.items()
        for name, (_, hp_type, bounds, _, idx2cat) in hparam_dict_items:
            if hp_type in {"real", "integer"}:
                real_value = real_vector[i]

                value = real_value
                if hp_type == "integer":
                    value = int(round(value))

                mixed_vector[i] = value
                i += 1
            else:
                n_categories = len(idx2cat.keys())
                subarray = real_vector[i:i + n_categories]
                max_idx = np.argmax(subarray)

                mixed_vector[i:i + n_categories] = 0.0
                mixed_vector[i + max_idx] = 1.0

                i += n_categories

        return mixed_vector


def create_ordered_dict(hp_dict: dict):
    ordered_dict = OrderedDict()

    HPInfo = namedtuple(
        typename="HPInfo",
        field_names=["name", "type", "bounds", "cat2idx", "idx2cat"]
    )

    for hp_name, (hp_type, hp_domain) in hp_dict.items():
        if hp_type == "categorical":
            # convert set to category2index dict
            cat2idx = OrderedDict()
            idx2cat = OrderedDict()

            for i, cat in enumerate(hp_domain):
                cat2idx[cat] = i
                idx2cat[i] = cat

            hp_info = HPInfo(hp_name, hp_type, None, cat2idx, idx2cat)
        else:
            hp_info = HPInfo(hp_name, hp_type, hp_domain, None, None)

        ordered_dict[hp_name] = hp_info

    return ordered_dict


def create_bounds_array(ordered_dict):
    bounds_list = list()

    for name, (_, hp_type, bounds, cat2idx, _) in ordered_dict.items():
        if hp_type in {"real", "integer"}:
            bounds_list.append([float(b) for b in bounds])
        else:
            for _ in range(len(cat2idx.keys())):
                bounds_list.append([0.0, 1.0])

    bounds_array = np.array(bounds_list)

    return bounds_array
