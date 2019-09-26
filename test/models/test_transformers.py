from collections import namedtuple

import numpy as np

from takepod.models import FeatureTransformer, TensorTransformer


def test_feature_transformer():
    mock_batch_class = namedtuple("Mock_feature_batch", ("mock_feature",))

    def mock_feature_extraction_fn(x):
        return x.mock_feature

    class MockTensorTransformer(TensorTransformer):

        def __init__(self):
            self.fit_called = False

        def fit(self,
                x: np.ndarray,
                y: np.ndarray):
            self.fit_called = True
            assert np.all(x == np.array([[1, 2], [3, 4]]))
            assert np.all(y == np.array([1, 2]))

        def transform(self,
                      x: np.array
                      ) -> np.ndarray:
            assert np.all(x == np.array([[4, 5], [6, 7]]))
            return np.array([3, 4])

    mock_tensor_transformer = MockTensorTransformer()
    feature_transformer = FeatureTransformer(mock_feature_extraction_fn,
                                             mock_tensor_transformer,
                                             requires_fitting=True)

    mock_feature_batch = mock_batch_class(mock_feature=np.array([[1, 2], [3, 4]]))
    y = np.array([1, 2])

    feature_transformer.fit(mock_feature_batch, y)
    assert mock_tensor_transformer.fit_called

    mock_feature_batch_2 = mock_batch_class(mock_feature=np.array([[4, 5], [6, 7]]))
    assert np.all(feature_transformer.transform(mock_feature_batch_2) == np.array([3, 4]))

    mock_tensor_transformer = MockTensorTransformer()
    feature_transformer = FeatureTransformer(mock_feature_extraction_fn,
                                             mock_tensor_transformer,
                                             requires_fitting=False)
    feature_transformer.fit(mock_feature_batch, y)
    assert not mock_tensor_transformer.fit_called
