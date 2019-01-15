"""Module contains fully connected neural network models."""
from sklearn.neural_network import MLPClassifier

from takepod.models.base_model import AbstractSupervisedModel


class ScikitMLPClassifier(AbstractSupervisedModel):
    """Simple scikitlearn multiperceptron model."""
    def __init__(self, **kwargs):
        self._model = MLPClassifier(max_iter=1, warm_start=True, **kwargs)

    def fit(self, X, y, **kwargs):
        self._model.fit(X=X, y=y)

    def predict(self, X, **kwargs):
        y_pred = self._model.predict(X=X)
