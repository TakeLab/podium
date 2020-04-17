"""Module contains svm models."""
import logging
from podium.models.model import AbstractSupervisedModel

_LOGGER = logging.getLogger(__name__)

try:
    from sklearn.svm import SVC, LinearSVC
except ImportError:
    _LOGGER.debug("Problem occured while trying to import sklearn. If the "
                  "library is not installed visit https://scikit-learn.org"
                  " for more details.")


class ScikitSVCModel(AbstractSupervisedModel):
    """Simple scikitlearn SVM model."""
    def __init__(self, **kwargs):
        self.reset(**kwargs)

    def reset(self, **kwargs):
        self._model = SVC(**kwargs)

    def fit(self, X, y, **kwargs):
        self._model.fit(X=X, y=y, **kwargs)

    def predict(self, X, **kwargs):
        y_pred = self._model.predict(X=X)
        return {AbstractSupervisedModel.PREDICTION_KEY: y_pred}


class ScikitLinearSVCModel(ScikitSVCModel):
    """Simple scikitlearn linear SVM model."""
    def reset(self, **kwargs):
        self._model = LinearSVC(**kwargs)
