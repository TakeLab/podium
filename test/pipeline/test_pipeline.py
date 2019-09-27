import pytest
import numpy as np

from takepod.pipeline import Pipeline
from takepod.storage import Field, ExampleFormat
from takepod.models import AbstractSupervisedModel


@pytest.fixture
def fields():
    name_dict = {
        "Marko": 1,
        "Darko": 2,
        "Ivana": 3
    }

    name_field = Field("Name", custom_numericalize=name_dict.get)
    score_field = Field("Score", tokenize=False, custom_numericalize=int)

    name_field.finalize()
    score_field.finalize()

    return {"Name": name_field,
            "Score": score_field}


mock_data = [
    ["Marko", 50],
    ["Darko", 60],
    ["Ivana", 45]
]


class MockModel:

    def fit(self, *args, **kwargs):
        pass

    def predict(self, x, **kwargs):
        return {AbstractSupervisedModel.PREDICTION_KEY: x}


class MockTrainer:
    def train(self, *args, **kwargs):
        pass


class MockFeatureTransformer:

    def transform(self, x_batch):
        return np.hstack((x_batch.Name, x_batch.Score))


def test_pipeline_from_raw(fields):

    # Test for list format
    fields_list = [fields['Name'], fields['Score']]
    list_pipeline = Pipeline(fields_list,
                             ExampleFormat.LIST,
                             feature_transformer=MockFeatureTransformer(),
                             model=MockModel())

    raw_list = ["Marko", 30]
    expected_prediction = np.array([[1, 30]])
    prediction = list_pipeline.predict_raw(raw_list)

    assert np.all(expected_prediction == prediction)

    fields_dict = {field.name: field for field in fields_list}
    dict_pipeline = Pipeline(fields_dict,
                             ExampleFormat.DICT,
                             feature_transformer=MockFeatureTransformer(),
                             model=MockModel())

    # Test for Dict format
    raw_dict = {'Name': "Marko", 'Score': 30}
    expected_prediction = np.array([[1, 30]])
    prediction = dict_pipeline.predict_raw(raw_dict)

    assert np.all(expected_prediction == prediction)

    # Test for csv
    raw_csv = "Marko,30"
    csv_pipeline = Pipeline(fields_list,
                            ExampleFormat.CSV,
                            feature_transformer=MockFeatureTransformer(),
                            model=MockModel())
    expected_prediction = np.array([[1, 30]])
    prediction = csv_pipeline.predict_raw(raw_csv)
    assert np.all(expected_prediction == prediction)
