"""
Module contains fully connected neural network models.
"""
from podium.experimental.models.model import AbstractSupervisedModel


try:
    from sklearn.neural_network import MLPClassifier
except ImportError:
    print(
        "Problem occured while trying to import sklearn. If the "
        "library is not installed visit https://scikit-learn.org "
        "for more details."
    )
    raise


class ScikitMLPClassifier(AbstractSupervisedModel):
    """
    Simple scikitlearn multiperceptron model.
    """

    def __init__(self, classes, **kwargs):
        """
        Constructor that initializes Sckit MLPClassfier with given list of
        classes.

        Parameters
        ----------
        classes : array, shape (n_classes)
            Classes across all calls to fit.
        """
        self.classes = classes
        self.reset(**kwargs)

    def reset(self, **kwargs):
        self._model = MLPClassifier(max_iter=1, **kwargs)

    def fit(self, X, y, **kwargs):
        """
        Method calls fit on multiperceptron model with given batch.

        It is supposed to be used as online learning.
        """
        self._model.partial_fit(X=X, y=y, classes=self.classes)

    def predict(self, X, **kwargs):
        y_pred = self._model.predict(X=X)
        return {AbstractSupervisedModel.PREDICTION_KEY: y_pred}
