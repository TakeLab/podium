import numpy as np
from takepod.models.fc_model import ScikitMLPClassifier

X = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]])
Y = np.array([0, 1, 0])


def test_scikit_mlp_model_shape():
    model = ScikitMLPClassifier(classes=np.unique(Y))
    model.fit(X=X, y=Y)
    result = model.predict(X=X)

    assert not result.get(model.PREDICTION_KEY, False) is False
    assert result.get(model.PREDICTION_KEY).shape == (3, )
