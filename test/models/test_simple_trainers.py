import pytest
import pytest_mock  # noqa

from takepod.models.simple_trainers import SimpleTrainer
from takepod.models.base_model import AbstractSupervisedModel
from takepod.storage.iterator import Iterator
from test.storage.conftest import (tabular_dataset, json_file_path)  # noqa


class MockSupervisedModel(AbstractSupervisedModel):
    def __init__(self):
        self.fit_num = 0
        self.predict_num = 0
        self.fit_init = False

    def fit(self, X, y, **kwargs):
        self.fit_num += 1
        self._check_fit_arguments(X, y, **kwargs)

    def predict(self, X, **kwargs):
        assert self.fit_init
        self.predict_num += 1

    @staticmethod
    def _check_fit_arguments(X, y, **kwargs):
        assert X.text.shape[0] == y.rating.shape[0]


@pytest.mark.usefixtures("tabular_dataset")  # noqa
def test_simple_trainer_no_num_epoch(tabular_dataset):
    iterator = Iterator(tabular_dataset, 1)
    model = MockSupervisedModel()
    with pytest.raises(ValueError):
        trainer = SimpleTrainer(model=model)
        trainer.train(iterator=iterator)


@pytest.mark.usefixtures("tabular_dataset")  # noqa
def test_simple_trainer_num_epoch(tabular_dataset):
    tabular_dataset.finalize_fields()
    iterator = Iterator(tabular_dataset, batch_size=len(tabular_dataset))
    model = MockSupervisedModel()
    trainer = SimpleTrainer(model=model)
    trainer.train(iterator=iterator, **{trainer.MAX_EPOCH_KEY: 10})
    assert model.fit_num == 10


def transform_fun(x, y):
    return x, y


@pytest.mark.usefixtures("tabular_dataset", "mocker")  # noqa
def test_simple_trainer_batch_transform_call(tabular_dataset, mocker):
    tabular_dataset.finalize_fields()
    iterator = Iterator(tabular_dataset, batch_size=len(tabular_dataset))

    with mocker.patch(
            "test.models.test_simple_trainers.transform_fun",
            return_value=next(iterator.__iter__())):
        model = MockSupervisedModel()
        trainer = SimpleTrainer(model=model)
        trainer.train(
            iterator=iterator,
            **{trainer.MAX_EPOCH_KEY: 10,
               SimpleTrainer.BATCH_TRANSFORM_FUN_KEY: transform_fun})
        assert transform_fun.call_count == 10
