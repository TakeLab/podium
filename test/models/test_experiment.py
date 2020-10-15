import pytest

from podium.models import Experiment, FeatureTransformer
from podium.datasets import Dataset
from podium.storage import Field, ExampleFactory, Vocab


def get_dataset():
    data = [{"Name": "Mark Dark",
             "Score": 5},
            {"Name": "Stephen Smith",
             "Score": 10},
            {"Name": "Ann Mann",
             "Score": 15}]

    name_field = Field("Name",
                       numericalizer=Vocab(),
                       keep_raw=True,
                       tokenizer="split")

    score_field = Field("Score",
                        numericalizer=int,
                        keep_raw=True,
                        tokenizer=None,
                        is_target=True)

    fields = {"Name": name_field,
              "Score": score_field}

    example_factory = ExampleFactory(fields)
    examples = [example_factory.from_dict(data_) for data_ in data]

    ds = Dataset(examples, fields)
    ds.finalize_fields()
    return ds


class MockDataset:
    pass


def mock_feature_transform_fun(x_batch):
    return x_batch.Score


def mock_label_transform_fun(y_batch):
    return y_batch.Score


class MockTransformer(FeatureTransformer):

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


class MockIterator:
    pass


@pytest.mark.parametrize("fit_transformer", (False, True))
def test_experiment_train(fit_transformer):
    test_dataset = get_dataset()

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

    class MockModel:
        def __init__(self, **kwargs):
            assert kwargs == expected_model_args

    class MockTrainer:

        def __init__(self):
            self.train_called = 0

        def train(self,
                  model,
                  dataset,
                  feature_transformer,
                  label_transform_fun,
                  **kwargs):
            assert isinstance(model, MockModel)
            assert dataset is test_dataset
            assert feature_transformer is mock_transformer
            assert label_transform_fun is mock_label_transform_fun
            assert kwargs == expected_trainer_args
            self.train_called += 1

    trainer = MockTrainer()

    experiment = Experiment(MockModel,
                            trainer=trainer,
                            feature_transformer=mock_transformer,
                            label_transform_fn=mock_label_transform_fun)

    experiment.set_default_model_args(**default_model_args)
    experiment.set_default_trainer_args(**default_trainer_args)

    experiment.fit(test_dataset,
                   model_args,
                   trainer_args)

    assert trainer.train_called == 1
    if fit_transformer:
        assert mock_transformer.fit_called == 1
    else:
        assert mock_transformer.fit_called == 0

# TODO test .predict()
