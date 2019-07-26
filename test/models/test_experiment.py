from collections import namedtuple

import numpy as np
from takepod.models import AbstractSupervisedModel, Experiment


class MockDataset:
    pass


def mock_batch_transform(x_batch, y_batch):
    return x_batch.input, y_batch.output


def test_experiment_train():
    default_model_args = {
        'm_arg1': 1,
        'm_arg2': 2
    }

    default_trainer_args = {
        't_arg1': 1,
        't_arg2': 2
    }

    model_args = {
        'm_arg2': 3,
        'm_arg3': 4
    }

    trainer_args = {
        't_arg2': 3,
        't_arg3': 4
    }

    expected_model_args = {
        'm_arg1': 1,
        'm_arg2': 3,
        'm_arg3': 4
    }

    expected_trainer_args = {
        't_arg1': 1,
        't_arg2': 3,
        't_arg3': 4
    }

    class MockIterator:
        input_batch_class = namedtuple("input_batch_class", ["input"])
        output_batch_class = namedtuple("output_batch_class", ["output"])

        def __iter__(self):
            x = np.array(
                [
                    [1, 2],
                    [3, 4]
                ])

            y = np.array([5, 6])

            input_batch = self.input_batch_class(input=x)
            target_batch = self.output_batch_class(output=y)
            yield input_batch, target_batch

    class MockModel:
        def __init__(self, **kwargs):
            assert kwargs == expected_model_args

    class MockTrainer:

        def __init__(self):
            self.train_called = False

        def train(self,
                  model,
                  iterator,
                  batch_transform=None,
                  **kwargs):
            assert isinstance(model, MockModel)
            assert isinstance(iterator, MockIterator)
            assert batch_transform is mock_batch_transform
            assert kwargs == expected_trainer_args
            self.train_called = True

    trainer = MockTrainer()

    experiment = Experiment(MockModel,
                            trainer,
                            mock_batch_transform,
                            lambda _: MockIterator())

    experiment.set_default_model_args(**default_model_args)
    experiment.set_default_trainer_args(**default_trainer_args)

    experiment.fit(MockDataset(),
                   model_args,
                   trainer_args)

    assert trainer.train_called


def test_experiment_predict():

    class MockIterator:
        input_batch_class = namedtuple("input_batch_class", ["input"])
        output_batch_class = namedtuple("output_batch_class", ["output"])

        def __iter__(self):
            x1 = np.array(
                [
                    [1, 2],
                    [3, 4]
                ])

            x2 = np.array(
                [
                    [7, 8],
                    [9, 10]
                ])

            input_batch1 = self.input_batch_class(input=x1)
            # Y values are just filler values for now
            target_batch1 = self.output_batch_class(output=np.arange(len(input_batch1)))
            yield input_batch1, target_batch1

            input_batch2 = self.input_batch_class(input=x2)
            target_batch2 = self.output_batch_class(output=np.arange(len(input_batch2)))
            yield input_batch2, target_batch2

    class MockModel:
        def __init__(self, **kwargs):
            self.expected_x_batches = [
                np.array(
                    [
                        [1, 2],
                        [3, 4]
                    ]),
                np.array(
                    [
                        [7, 8],
                        [9, 10]
                    ])
            ]

            self.predictions = [
                np.array([5, 6]),
                np.array([11, 12])
            ]

            self.current_batch = 0

        def predict(self, x):
            # Check if correct batches received
            assert np.all(x == self.expected_x_batches[self.current_batch])
            y = self.predictions[self.current_batch]
            self.current_batch += 1

            return {AbstractSupervisedModel.PREDICTION_KEY: y}

    class MockTrainer:
        def train(self, model, iterator, batch_transform=None, **kwargs):
            pass

    experiment = Experiment(
        MockModel,
        MockTrainer(),
        mock_batch_transform,
        lambda _: MockIterator(),
        lambda _: MockIterator()
    )
    experiment.fit(MockDataset())
    y = experiment.predict(MockDataset())

    assert np.all(y == np.array([[5], [6], [11], [12]]))
