import pytest
import pytest_mock  # noqa

from podium.models.impl.simple_trainers import SimpleTrainer
from podium.models.model import AbstractSupervisedModel
from podium.models import FeatureTransformer
from podium.datasets.iterator import Iterator

from ..storage.conftest import (tabular_dataset, json_file_path) # noqa


@pytest.fixture
@pytest.mark.usefixtures("mocker")
def model(mocker):
    model = mocker.MagicMock(spec=AbstractSupervisedModel)
    return model


@pytest.mark.usefixtures("tabular_dataset", "model")  # noqa
def test_simple_trainer_num_epoch(tabular_dataset, model):  # noqa
    tabular_dataset.finalize_fields()
    iterator = Iterator(batch_size=len(tabular_dataset))
    trainer = SimpleTrainer()
    feature_transformer = FeatureTransformer(lambda x: x)
    trainer.train(model=model,
                  dataset=tabular_dataset,
                  iterator=iterator,
                  feature_transformer=feature_transformer,
                  label_transform_fun=lambda y: y,
                  **{trainer.MAX_EPOCH_KEY: 10})
    assert model.fit.call_count == 10


def mock_feature_transform_fun(x):
    return x


def mock_label_transform_fun(y):
    return y

@pytest.mark.usefixtures("tabular_dataset", "mocker", "model")  # noqa
def test_simple_trainer_batch_transform_call(tabular_dataset, mocker, model):  # noqa
    tabular_dataset.finalize_fields()
    iterator = Iterator(tabular_dataset, batch_size=len(tabular_dataset))

    mocker.patch("tests.models.test_simple_trainers.mock_feature_transform_fun",
                 return_value=next(iterator.__iter__())[0])
    mocker.patch("tests.models.test_simple_trainers.mock_label_transform_fun",
                 return_value=next(iterator.__iter__())[1])

    feature_transformer = FeatureTransformer(mock_feature_transform_fun)
    trainer = SimpleTrainer()
    trainer.train(
        model=model,
        dataset=tabular_dataset,
        iterator=iterator,
        feature_transformer=feature_transformer,
        label_transform_fun=mock_label_transform_fun,
        **{trainer.MAX_EPOCH_KEY: 10})
    assert mock_feature_transform_fun.call_count == 10  # pylint: disable=E1101
    assert mock_label_transform_fun.call_count == 10  # pylint: disable=E1101
