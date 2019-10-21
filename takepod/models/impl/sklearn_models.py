"""Module contains a collection of sklearn models."""
import logging
from enum import Enum
from takepod.models.model import AbstractSupervisedModel

_LOGGER = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC
except ImportError as ex:
    _LOGGER.debug("Problem occured while trying to import sklearn. If the "
                  "library is not installed visit https://scikit-learn.org"
                  " for more details.")


class ModelType(Enum):
    STOCHASTIC_GRADIENT_DESCENT = 'sgd'
    LOGISTIC_REGRESSION = 'log'
    SUPPORT_VECTOR_MACHINE = 'svm'


class SklearnModels(AbstractSupervisedModel):
    """Simple scikitlearn multiperceptron model."""
    def __init__(self, **kwargs):
        """Constructor that initializes Scikit models
        that work with entire datasets.

        """
        model_type = kwargs['model']['model_type']
        model_args = kwargs['model']['model_specific']

        if model_type == ModelType.STOCHASTIC_GRADIENT_DESCENT: 
            self._model = SGDClassifier(**model_args)
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            self._model = LogisticRegression(**model_args)
        elif model_type == ModelType.SUPPORT_VECTOR_MACHINE:
            self._model = SVC(**model_args)
        else:
            raise ValueError(
                "Unknown model type provided {}, supported models: {}".format(
                    model_type, [e.value for e in ModelType]
                )
            )

    def reset(self, **kwargs):
        pass

    def fit(self, X, y, **kwargs):
        """Method calls fit on multiperceptron model with given batch.
        It is supposed to be used as online learning.
        """
        self._model.fit(X=X, y=y)

    def predict(self, X, **kwargs):
        y_pred = self._model.predict(X=X)
        return {AbstractSupervisedModel.PREDICTION_KEY: y_pred}

