from collections import namedtuple

import numpy as np

from takepod.models import FeatureTransformer, TensorTransformer, \
    SklearnTensorTransformerWrapper


class MockTensorTransformer(TensorTransformer):

    def __init__(self, requires_fitting):
        self.requires_fitting_flag = requires_fitting

    def fit(self,
            x: np.ndarray,
            y: np.ndarray):
        pass

    def transform(self,
                  x: np.array
                  ) -> np.ndarray:
        return [3, 4]

    def requires_fitting(self) -> bool:
        return self.requires_fitting_flag


def test_feature_transformer(mocker):
    mock_batch_class = namedtuple("Mock_feature_batch", ("mock_feature",))

    def mock_feature_extraction_fn(x):
        return x.mock_feature

    mock_tensor_transformer = MockTensorTransformer(requires_fitting=True)

    mocker.spy(mock_tensor_transformer, 'fit')
    mocker.spy(mock_tensor_transformer, 'transform')

    feature_transformer = FeatureTransformer(mock_feature_extraction_fn,
                                             mock_tensor_transformer)

    mock_feature_batch = mock_batch_class(mock_feature=[1, 2])
    y = [3, 4]

    feature_transformer.fit(mock_feature_batch, y)
    mock_tensor_transformer.fit.assert_called_once_with([1, 2], [3, 4])

    mock_feature_batch_2 = mock_batch_class(mock_feature=[4, 5])
    assert np.all(feature_transformer.transform(mock_feature_batch_2) == [3, 4])
    mock_tensor_transformer.transform.assert_called_once_with([4, 5])

    mock_tensor_transformer_no_fit = MockTensorTransformer(requires_fitting=False)
    mocker.spy(mock_tensor_transformer_no_fit, 'fit')
    mocker.spy(mock_tensor_transformer_no_fit, 'transform')
    feature_transformer = FeatureTransformer(mock_feature_extraction_fn,
                                             mock_tensor_transformer_no_fit)

    feature_transformer.fit(mock_feature_batch, y)
    mock_tensor_transformer_no_fit.fit.assert_not_called()
    assert np.all(feature_transformer.transform(mock_feature_batch_2) == [3, 4])
    mock_tensor_transformer_no_fit.transform.assert_called_once_with([4, 5])


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
