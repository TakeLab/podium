import numpy as np

from takepod.pipeline import Pipeline
from takepod.storage import Field, ExampleFormat
from takepod.models import AbstractSupervisedModel, FeatureTransformer
from takepod.datasets import SingleBatchIterator

name_dict = {
    "Marko": 1,
    "Darko": 2,
    "Ivana": 3
}

mock_data = [
    ["Marko", 50, 20],
    ["Darko", 60, 30],
    ["Ivana", 45, 40]
]


def get_fields():
    name_field = Field("Name", custom_numericalize=name_dict.get)
    score_field = Field("Score", tokenize=False, custom_numericalize=int)
    age_field = Field("Age", tokenize=False, custom_numericalize=int, is_target=True)

    name_field.finalize()
    score_field.finalize()

    return {"Name": name_field,
            "Score": score_field,
            "Age": age_field}


def mock_feature_transform(x_batch):
    return np.hstack((x_batch.Name, x_batch.Score))


class MockFeatureTransformer(FeatureTransformer):

    def transform(self, x_batch):
        return np.hstack((x_batch.Name, x_batch.Score))

    def requires_fitting(self):
        return False


def mock_label_extractor(y_batch):
    return y_batch.Age


def test_pipeline_predict_raw():
    class MockModel:

        def fit(self, x, y, **kwargs):
            pass

        def predict(self, x, **kwargs):
            return {AbstractSupervisedModel.PREDICTION_KEY: x}

    fields = get_fields()

    # Test for list format
    fields_list = [fields['Name'], fields['Score'], None]
    list_pipeline = Pipeline(fields_list,
                             example_format="list",
                             feature_transformer=mock_feature_transform,
                             model=MockModel())

    raw_list = ["Marko", 30]
    expected_prediction = np.array([1, 30])
    prediction = list_pipeline.predict_raw(raw_list)

    assert np.all(expected_prediction == prediction)

    fields_dict = {field.name: field for field in fields_list if field}
    dict_pipeline = Pipeline(fields_dict,
                             ExampleFormat.DICT,
                             feature_transformer=mock_feature_transform,
                             model=MockModel())

    # Test for Dict format
    raw_dict = {'Name': "Marko", 'Score': 30}
    expected_prediction = np.array([1, 30])
    prediction = dict_pipeline.predict_raw(raw_dict)

    assert np.all(expected_prediction == prediction)

    # Test for csv
    raw_csv = "Marko,30"
    csv_pipeline = Pipeline(fields_list,
                            ExampleFormat.CSV,
                            feature_transformer=mock_feature_transform,
                            model=MockModel())
    expected_prediction = np.array([1, 30])
    prediction = csv_pipeline.predict_raw(raw_csv)
    assert np.all(expected_prediction == prediction)


def test_pipeline_fit_raw():
    fields = get_fields()

    class MockModel:

        def fit(self, x_batch, y_batch):
            expected_x_batch = np.array(
                [[1, 50],
                 [2, 60],
                 [3, 45]]
            )
            expected_y_batch = np.array(
                [[20],
                 [30],
                 [40]]
            )

            assert np.all(x_batch == expected_x_batch)
            assert np.all(y_batch == expected_y_batch)

    class MockTrainer:

        def train(self,
                  model,
                  dataset,
                  feature_transformer,
                  label_transform_fun,
                  **kwargs):
            #  Using single batch iterator so only one batch
            x_batch, y_batch = dataset.batch()
            model.fit(feature_transformer.transform(x_batch),
                      label_transform_fun(y_batch))

    # Test for list format
    fields_list = [fields['Name'], fields['Score'], fields['Age']]

    list_pipeline = Pipeline(fields_list,
                             model=MockModel,
                             trainer=MockTrainer(),
                             example_format="list",
                             feature_transformer=mock_feature_transform,
                             label_transform_fn=mock_label_extractor)

    list_pipeline.fit_raw(mock_data)
    list_pipeline.partial_fit_raw(mock_data)


def test_output_transform_fn():
    class MockModel:

        def fit(self, x, y, **kwargs):
            pass

        def predict(self, x, **kwargs):
            return {AbstractSupervisedModel.PREDICTION_KEY: x}

    transform_dict = {val: key.upper() for key, val in name_dict.items()}

    fields = get_fields()
    fields_list = [fields['Name'], fields['Score']]
    list_pipeline = Pipeline(fields_list,
                             ExampleFormat.LIST,
                             feature_transformer=MockFeatureTransformer(),
                             model=MockModel(),
                             output_transform_fn=lambda x: transform_dict[x[0]])

    for example in mock_data:
        assert list_pipeline.predict_raw(example) == example[0].upper()
