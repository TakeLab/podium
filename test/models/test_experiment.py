from collections import namedtuple

import numpy as np
from takepod.models import Experiment


class MockDataset:
    pass


def mockBatchTransform(x_batch, y_batch):
    return x_batch.input, y_batch.output


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
        return input_batch, target_batch


def test_experiment():
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

    class MockModel:
        def __init__(self, **kwargs):
            assert kwargs == expected_model_args

    class MockTrainer:
        def train(self,
                  model,
                  iterator,
                  batch_transform=None,
                  **kwargs):
            assert isinstance(model, MockModel)
            assert isinstance(iterator, MockIterator)
            assert batch_transform is mockBatchTransform
            assert kwargs == expected_trainer_args

    experiment = Experiment(MockModel,
                            MockTrainer(),
                            mockBatchTransform,
                            lambda _: MockIterator())

    experiment.set_default_model_args(**default_model_args)
    experiment.set_default_trainer_args(**default_trainer_args)

    experiment.fit(MockDataset(),
                   model_args,
                   trainer_args)
