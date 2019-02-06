"""Module contains fully connected neural network models."""
from sklearn.neural_network import MLPClassifier

from takepod.models.base_model import AbstractSupervisedModel


class ScikitMLPClassifier(AbstractSupervisedModel):
    """Simple scikitlearn multiperceptron model."""
    def __init__(self, classes, **kwargs):
        """Constructor that initializes Sckit MLPClassfier with given list of
        classes.

        Parameters
        ----------
        classes : array, shape (n_classes)
            Classes across all calls to fit.
        """
        self._model = MLPClassifier(max_iter=1, **kwargs)
        self.classes = classes

    def fit(self, X, y, **kwargs):
        self._model.partial_fit(X=X, y=y, classes=self.classes)

    def predict(self, X, **kwargs):
        y_pred = self._model.predict(X=X)
        return {AbstractSupervisedModel.PREDICTION_KEY: y_pred}
