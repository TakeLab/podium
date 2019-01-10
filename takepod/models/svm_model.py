from sklearn.svm import SVC
from takepod.models.base_model import AbstractSupervisedModel


class ScikitSVCModel(AbstractSupervisedModel):
    def __init__(self, **kwargs):
        self._model = SVC(**kwargs)

    def fit(self, X, y, **kwargs):
        self._model.fit(X=X, y=y, **kwargs)

    def predict(self, X, **kwargs):
        y_pred = self._model.predict(X=X)
