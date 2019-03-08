from takepod.models import AbstractSupervisedModel
from takepod.models.blcc.BiLSTM import BiLSTM


class BLCCModel(AbstractSupervisedModel):

    def __init__(self, **kwargs) -> None:
        self.model = BiLSTM(**kwargs)

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X)
