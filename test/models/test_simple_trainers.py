import pytest
import pytest_mock  # noqa

from takepod.models.simple_trainers import SimpleTrainer
from takepod.models.base_model import AbstractSupervisedModel
from takepod.storage.iterator import Iterator
from test.storage.conftest import (tabular_dataset, json_file_path)  # noqa


@pytest.fixture()
@pytest.mark.usefixtures("mocker")
def model(mocker):
    model = mocker.MagicMock(spec=AbstractSupervisedModel)
    return model


@pytest.mark.usefixtures("tabular_dataset", "model")  # noqa
def test_simple_trainer_no_num_epoch(tabular_dataset, model):
    iterator = Iterator(tabular_dataset, 1)
    with pytest.raises(ValueError):
        trainer = SimpleTrainer(model=model)
        trainer.train(iterator=iterator)


@pytest.mark.usefixtures("tabular_dataset", "model")  # noqa
def test_simple_trainer_num_epoch(tabular_dataset, model):
    tabular_dataset.finalize_fields()
    iterator = Iterator(tabular_dataset, batch_size=len(tabular_dataset))
    trainer = SimpleTrainer(model=model)
    trainer.train(iterator=iterator, **{trainer.MAX_EPOCH_KEY: 10})
    assert model.fit.call_count == 10


def _transform_fun(x, y):
    return x, y


@pytest.mark.usefixtures("tabular_dataset", "mocker", "model")  # noqa
def test_simple_trainer_batch_transform_call(tabular_dataset, mocker, model):
    tabular_dataset.finalize_fields()
    iterator = Iterator(tabular_dataset, batch_size=len(tabular_dataset))

    with mocker.patch(
            "test.models.test_simple_trainers._transform_fun",
            return_value=next(iterator.__iter__())):
        trainer = SimpleTrainer(model=model)
        trainer.train(
            iterator=iterator,
            **{trainer.MAX_EPOCH_KEY: 10,
               SimpleTrainer.BATCH_TRANSFORM_FUN_KEY: _transform_fun})
        assert _transform_fun.call_count == 10  # pylint: disable=E1101
