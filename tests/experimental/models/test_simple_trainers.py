import pytest
import pytest_mock  # noqa

from podium.datasets.iterator import Iterator
from podium.experimental.models import FeatureTransformer
from podium.experimental.models.impl.simple_trainers import SimpleTrainer
from podium.experimental.models.model import AbstractSupervisedModel

from ...conftest import json_file_path, tabular_dataset  # noqa


@pytest.fixture
@pytest.mark.usefixtures("mocker")
def model(mocker):
    model = mocker.MagicMock(spec=AbstractSupervisedModel)
    return model


@pytest.mark.usefixtures("tabular_dataset", "model")  # noqa
def test_simple_trainer_num_epoch(tabular_dataset, model):  # noqa
    iterator = Iterator(batch_size=len(tabular_dataset))
    trainer = SimpleTrainer()
    feature_transformer = FeatureTransformer(lambda x: x)
    trainer.train(
        model=model,
        dataset=tabular_dataset,
        iterator=iterator,
        feature_transformer=feature_transformer,
        label_transform_fun=lambda y: y,
        **{trainer.MAX_EPOCH_KEY: 10},
    )
    assert model.fit.call_count == 10


def mock_feature_transform_fun(x):
    return x


def mock_label_transform_fun(y):
    return y


@pytest.mark.usefixtures("tabular_dataset", "mocker", "model")  # noqa
def test_simple_trainer_batch_transform_call(tabular_dataset, mocker, model):  # noqa
    iterator = Iterator(tabular_dataset, batch_size=len(tabular_dataset))

    mocker.patch(
        "tests.experimental.models.test_simple_trainers.mock_feature_transform_fun",
        return_value=next(iter(iterator))[0],
    )
    mocker.patch(
        "tests.experimental.models.test_simple_trainers.mock_label_transform_fun",
        return_value=next(iter(iterator))[1],
    )

    feature_transformer = FeatureTransformer(mock_feature_transform_fun)
    trainer = SimpleTrainer()
    trainer.train(
        model=model,
        dataset=tabular_dataset,
        iterator=iterator,
        feature_transformer=feature_transformer,
        label_transform_fun=mock_label_transform_fun,
        **{trainer.MAX_EPOCH_KEY: 10},
    )
    assert mock_feature_transform_fun.call_count == 10  # pylint: disable=E1101
    assert mock_label_transform_fun.call_count == 10  # pylint: disable=E1101
