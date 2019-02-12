import numpy as np
from takepod.models.svm_model import ScikitSVCModel

X = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]])
Y = np.array([0, 1, 0])


def test_scikit_svc_model_shape():
    svc = ScikitSVCModel(gamma='auto')
    svc.fit(X=X, y=Y)
    result = svc.predict(X=X)

    assert not result.get(svc.PREDICTION_KEY, False) is False
    assert result.get(svc.PREDICTION_KEY).shape == (3, )
