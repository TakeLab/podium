from collections import namedtuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from takepod.models import FeatureTransformer, TensorTransformer, SklearnTensorTransformerWrapper


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

        def requires_fitting(self) -> bool:
            return True

    class MockTensorTransformerNoFitting(MockTensorTransformer):

        def requires_fitting(self):
            return False

    mock_tensor_transformer = MockTensorTransformer()
    feature_transformer = FeatureTransformer(mock_feature_extraction_fn,
                                             mock_tensor_transformer)

    mock_feature_batch = mock_batch_class(mock_feature=np.array([[1, 2], [3, 4]]))
    y = np.array([1, 2])

    feature_transformer.fit(mock_feature_batch, y)
    assert mock_tensor_transformer.fit_called

    mock_feature_batch_2 = mock_batch_class(mock_feature=np.array([[4, 5], [6, 7]]))
    assert np.all(feature_transformer.transform(mock_feature_batch_2) == np.array([3, 4]))

    mock_tensor_transformer_no_fit = MockTensorTransformerNoFitting()
    feature_transformer = FeatureTransformer(mock_feature_extraction_fn,
                                             mock_tensor_transformer_no_fit)
    feature_transformer.fit(mock_feature_batch, y)
    assert not mock_tensor_transformer_no_fit.fit_called
    assert np.all(feature_transformer.transform(mock_feature_batch_2) == np.array([3, 4]))


def test_sklearn_feature_transformer_wrapper(mocker):
    class MockSklearnTransformer:

        def fit(self, x, y):
            pass

        def transform(self, x):
            return x + 1

    # test with fitting
    tensor_transformer = MockSklearnTransformer()
    mocker.spy(tensor_transformer, 'fit')
    mocker.spy(tensor_transformer, 'transform')

    mock_feature_batch = np.array([[1, 2, 3]])
    mock_label_batch = np.array([[2, 3, 4]])

    wrapper = SklearnTensorTransformerWrapper(tensor_transformer, requires_fitting=True)

    assert wrapper.requires_fitting()

    wrapper.fit(mock_feature_batch, mock_label_batch)
    tensor_transformer.fit.assert_called_once_with(mock_feature_batch, mock_label_batch)

    result = tensor_transformer.transform(mock_feature_batch)
    tensor_transformer.transform.assert_called_once_with(mock_feature_batch)
    assert np.all(result == mock_feature_batch + 1)

    # test without fitting
    tensor_transformer = MockSklearnTransformer()
    mocker.spy(tensor_transformer, 'fit')
    mocker.spy(tensor_transformer, 'transform')

    wrapper = SklearnTensorTransformerWrapper(tensor_transformer, requires_fitting=False)

    assert not wrapper.requires_fitting()

    wrapper.fit(mock_feature_batch, mock_label_batch)
    tensor_transformer.fit.assert_not_called()

    result = tensor_transformer.transform(mock_feature_batch)
    tensor_transformer.transform.assert_called_once_with(mock_feature_batch)
    assert np.all(result == mock_feature_batch + 1)