import numpy as np
from unittest.mock import Mock

from podium.pipeline import Pipeline
from podium.storage import (
    Field, ExampleFormat,
    MultioutputField,
    LabelField, TokenizedField
)
from podium.models import AbstractSupervisedModel, FeatureTransformer


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
    name_field = Field("Name", custom_numericalize=name_dict.get, keep_raw=True)
    score_field = Field("Score", tokenizer=None, custom_numericalize=int,
                        keep_raw=True)
    age_field = LabelField("Age", custom_numericalize=int)

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
                             feature_transformer=mock_feature_transform,
                             model=MockModel(),
                             output_transform_fn=lambda x: transform_dict[x[0]])

    for example in mock_data:
        assert list_pipeline.predict_raw(example) == example[0].upper()


def test_pipeline_multioutputfield_without_target():
    # case where name is used to generate two fields => name and case
    name_field = Field("Name", custom_numericalize=name_dict.get, keep_raw=True)
    case_field = Field("Case", custom_numericalize={True: 1, False: 0}.get,
                       keep_raw=True)

    def get_case(raw, tokenized):
        return raw, list(map(str.islower, tokenized))

    case_field.add_posttokenize_hook(get_case)
    age_field = LabelField("Age", custom_numericalize=int)

    name_field.finalize()
    case_field.finalize()

    model = Mock()
    model.predict.return_value = {
        AbstractSupervisedModel.PREDICTION_KEY: [0]
    }
    input_field = MultioutputField((name_field, case_field))
    pipeline = Pipeline(
        [input_field, age_field],
        ExampleFormat.LIST,
        feature_transformer=lambda x: np.hstack((x.Name, x.Case)),
        model=model,
    )
    assert pipeline.predict_raw(["Marko"]) == 0
    assert pipeline.feature_fields == [input_field]
    # 1 -> Marko, 0 -> False
    assert all(model.predict.call_args[0][0][0] == [1, 0])


def test_pipeline_multioutputfield_with_some_target():
    mask_dict = {"XXXXX": 1, "XXXX": 2}
    text_dict = {"Marko": 1, "radi": 2}
    mask_field = Field(
        "Masked", custom_numericalize=mask_dict.get, keep_raw=True
    )
    text_field = Field(
        "Text", custom_numericalize=text_dict.get, is_target=True,
        keep_raw=True
    )
    text_field.finalize()

    model = Mock()
    model.predict.return_value = {
        AbstractSupervisedModel.PREDICTION_KEY: [0]
    }
    field = MultioutputField((mask_field, text_field))
    pipeline = Pipeline(
        # multioutput fields can be used for non-target
        [field],
        ExampleFormat.LIST,
        feature_transformer=lambda x: x.Masked,
        model=model,
    )
    assert pipeline.predict_raw(["XXXXX XXXX"]) == 0
    assert pipeline.feature_fields == [field]
    # 1 -> Marko, 0 -> False
    assert all(model.predict.call_args[0][0][0] == [1, 2])


def test_pipeline_multioutputfield_with_all_targets():
    text_dict = {"Marko": 1, "radi": 2}
    mask_dict = {"XXXXX": 1, "XXXX": 2}
    mask_field = Field("Mask", custom_numericalize=mask_dict.get)
    text1_field = Field(
        "Text1", custom_numericalize=text_dict.get,
        is_target=True, keep_raw=True
    )
    text2_field = Field(
        "Text2", custom_numericalize=text_dict.get,
        is_target=True, keep_raw=True
    )
    mask_field.finalize()

    model = Mock()
    model.predict.return_value = {
        AbstractSupervisedModel.PREDICTION_KEY: [0]
    }

    pipeline = Pipeline(
        # multioutput fields can be used for non-target
        [mask_field, MultioutputField((text1_field, text2_field))],
        ExampleFormat.LIST,
        feature_transformer=lambda x: x.Mask,
        model=model,
    )
    assert pipeline.predict_raw(["XXXXX XXXX"]) == 0
    assert pipeline.feature_fields == [mask_field]
    # 1 -> Marko, 0 -> False
    assert all(model.predict.call_args[0][0][0] == [1, 2])


def test_pipeline_nested_fields_no_targets():
    name_field = Field(
        "Name", custom_numericalize=name_dict.get,
        keep_raw=True
    )
    case_field = Field(
        "Case", custom_numericalize={True: 1, False: 0}.get,
        keep_raw=True
    )

    def get_case(raw, tokenized):
        return raw, list(map(str.islower, tokenized))

    case_field.add_posttokenize_hook(get_case)
    age_field = Field(
        "Age", tokenizer=None, custom_numericalize=int, is_target=True
    )

    name_field.finalize()

    model = Mock()
    model.predict.return_value = {
        AbstractSupervisedModel.PREDICTION_KEY: [0]
    }
    pipeline = Pipeline(
        # nested fields also can contain target fields
        ((name_field, case_field), age_field),
        ExampleFormat.LIST,
        feature_transformer=lambda x: np.hstack((x.Name, x.Case)),
        model=model,
    )
    assert pipeline.predict_raw(["Marko"]) == 0
    assert pipeline.feature_fields == [(name_field, case_field)]
    # 1 -> Marko, 0 -> False
    assert all(model.predict.call_args[0][0][0] == [1, 0])


def test_pipeline_nested_fields_all_targets():
    text_dict = {"Marko": 1, "radi": 2}
    mask_dict = {"XXXXX": 1, "XXXX": 2}
    text_field = Field(
        "Text", custom_numericalize=text_dict.get,
        keep_raw=True
    )
    mask1_field = Field(
        "Masked1", custom_numericalize=mask_dict.get, is_target=True,
        keep_raw=True
    )
    mask2_field = Field(
        "Masked2", custom_numericalize=mask_dict.get, is_target=True,
        keep_raw=True
    )
    text_field.finalize()

    model = Mock()
    model.predict.return_value = {
        AbstractSupervisedModel.PREDICTION_KEY: [0]
    }
    pipeline = Pipeline(
        # multioutput fields can be used for non-target
        [text_field, ((mask1_field, mask2_field))],
        ExampleFormat.LIST,
        feature_transformer=lambda x: x.Text,
        model=model,
    )
    assert pipeline.predict_raw(["Marko"]) == 0
    assert pipeline.feature_fields == [text_field]
    assert all(model.predict.call_args[0][0][0] == [1])


def test_pipeline_label_and_tokenized_fields():
    # case where name is used to generate two fields => name and case
    name_field = TokenizedField("Name", custom_numericalize=name_dict.get)
    case_field = TokenizedField("Case", custom_numericalize={True: 1, False: 0}.get)

    def get_case(raw, tokenized):
        return raw, list(map(str.islower, tokenized))
    case_field.add_posttokenize_hook(get_case)

    age_field = LabelField(
        "Age", custom_numericalize=int,
    )

    name_field.finalize()
    case_field.finalize()

    model = Mock()
    model.predict.return_value = {
        AbstractSupervisedModel.PREDICTION_KEY: [0]
    }

    pipeline = Pipeline(
        # nested fields also can contain target fields
        ((name_field, case_field), age_field),
        ExampleFormat.LIST,
        feature_transformer=lambda x: np.hstack((x.Name, x.Case)),
        model=model,
    )
    assert pipeline.predict_raw(["Marko", "vrago"]) == 0
