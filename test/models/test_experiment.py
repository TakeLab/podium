from collections import namedtuple

import pytest
import numpy as np
from takepod.models import AbstractSupervisedModel, Experiment
from takepod.datasets import Dataset, Iterator
from takepod.storage import Field, ExampleFactory, Vocab


@pytest.fixture
def dataset():
    data = [{"Name": "Mark Dark",
             "Score": 5},
            {"Name": "Stephen Smith",
             "Score": 10},
            {"Name": "Ann Mann",
             "Score": 15}]

    name_field = Field("Name",
                       vocab=Vocab(),
                       store_as_raw=True,
                       tokenizer="split")

    score_field = Field("Score",
                        custom_numericalize=int,
                        tokenize=False,
                        is_target=True)

    fields = {"Name": name_field,
              "Score": score_field}

    example_factory = ExampleFactory(fields)
    examples = [example_factory.from_dict(data_) for data_ in data]

    ds = Dataset(examples, fields)
    ds.finalize_fields()
    return ds

def MockDataset():
    pass

def mock_feature_transform_fun(x_batch):
    return x_batch.Score


def mock_label_transform_fun(y_batch):
    return y_batch.Score


class MockTransformer:

    def __init__(self, to_fit):
        self.to_fit = to_fit
        self.fit_called = 0

    def fit(self, x, y):
        self.fit_called += 1
        pass

    def transform(self, x_batch):
        return mock_feature_transform_fun(x_batch)

    def requires_fitting(self):
        return self.to_fit


@pytest.mark.parametrize("fit_transformer", (False, True))
def test_experiment_train(dataset, fit_transformer):
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

    mock_transformer = MockTransformer(fit_transformer)

    my_iterator = Iterator(dataset)

    class MockModel:
        def __init__(self, **kwargs):
            assert kwargs == expected_model_args

    class MockTrainer:

        def __init__(self):
            self.train_called = 0

        def train(self,
                  model,
                  iterator,
                  feature_transformer,
                  label_transform_fun,
                  **kwargs):
            assert isinstance(model, MockModel)
            assert iterator is my_iterator
            assert feature_transformer is mock_transformer
            assert label_transform_fun is mock_label_transform_fun
            assert kwargs == expected_trainer_args
            self.train_called += 1

    trainer = MockTrainer()

    experiment = Experiment(MockModel,
                            trainer=trainer,
                            training_iterator_callable=lambda _: my_iterator,
                            feature_transformer=mock_transformer,
                            label_transform_fun=mock_label_transform_fun)

    experiment.set_default_model_args(**default_model_args)
    experiment.set_default_trainer_args(**default_trainer_args)

    experiment.fit(dataset,
                   model_args,
                   trainer_args)

    assert trainer.train_called == 1
    if fit_transformer:
        assert mock_transformer.fit_called == 1
    else:
        assert mock_transformer.fit_called == 0


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
                np.array([[5], [6]]),
                np.array([[11], [12]])
            ]

            self.current_batch = 0

        def predict(self, x):
            # Check if correct batches received
            assert np.all(x == self.expected_x_batches[self.current_batch])
            y = self.predictions[self.current_batch]
            self.current_batch += 1

            return {AbstractSupervisedModel.PREDICTION_KEY: y}

    class MockTrainer:
        def train(self, model, iterator, feature_transform_fun=None,
                  label_transform_fun=None, **kwargs):
            pass

    experiment = Experiment(
        MockModel,
        trainer=MockTrainer(),
        training_iterator_callable=lambda _: MockIterator(),
        prediction_iterator_callable=lambda _: MockIterator()

    )
    experiment.fit(MockDataset())
    y = experiment.predict(MockDataset())

    assert np.all(y == np.array([[5], [6], [11], [12]]))
