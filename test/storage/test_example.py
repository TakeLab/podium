from takepod.storage.example import Example
import pytest


class MockField:
    def __init__(self, name):
        self.name = name

    def preprocess(self, data):
        return data, [data]


@pytest.mark.parametrize(
    "data_dict, fields_dict",
    [
        (
                {"text": "this is a review", "rating": 4.5, "sentiment": 1},
                {"text": (MockField("words"), MockField("chars")),
                 "rating": MockField("label"),
                 "sentiment": (MockField("sentiment"), MockField("polarity"))}
        ),
        (
                {"x": "data"},
                {"x": (MockField("docs"), MockField("paragraphs"),
                       MockField("sents"), MockField("words"),
                       MockField("syllables"), MockField("chars"))
                 }
        ),
    ]
)
def test_fromdict_ok(data_dict, fields_dict):
    ex = Example.fromdict(data_dict, fields_dict)

    for data_key, data_val in data_dict.items():
        fields = fields_dict[data_key]

        if not isinstance(fields, tuple):
            fields = (fields,)

        for field in fields:
            assert ex.__getattribute__(field.name)[0] == data_dict[data_key]
            assert ex.__getattribute__(field.name)[1][0] == data_dict[data_key]


@pytest.mark.parametrize(
    "data_dict, fields_dict, exception_type",
    [
        (
                {"text": "this is a review", "rating": 4.5, "sentiment": 1},
                {"not_text": (MockField("words"), MockField("chars")),
                 "rating": MockField("label"),
                 "sentiment": (MockField("sentiment"), MockField("polarity"))},
                ValueError
        ),
    ]
)
def test_fromdict_exception(data_dict, fields_dict, exception_type):
    try:
        Example.fromdict(data_dict, fields_dict)
    except Exception as e:
        t = type(e)

    assert t == exception_type


@pytest.mark.parametrize(
    "json_data, fields_dict, expected_data_dict",
    [
        (
                '{"text": "this is a review", "rating": 4.5, "sentiment": 1}',
                {"text": (MockField("words"), MockField("chars")),
                 "rating": MockField("label"),
                 "sentiment": (MockField("sentiment"), MockField("polarity"))},
                {"text": "this is a review", "rating": 4.5, "sentiment": 1},
        ),
        (
                '{"x": "data"}',
                {"x": (
                        MockField("docs"), MockField("paragraphs"),
                        MockField("sents"),
                        MockField("words"),
                        MockField("syllables"), MockField("chars"))},
                {"x": "data"},
        ),
    ]
)
def test_fromJSON_ok(json_data, fields_dict, expected_data_dict):
    ex = Example.fromJSON(json_data, fields_dict)

    for data_key, data_val in expected_data_dict.items():
        fields = fields_dict[data_key]

        if not isinstance(fields, tuple):
            fields = (fields,)

        for field in fields:
            assert ex.__getattribute__(field.name)[0] == expected_data_dict[
                data_key]
            assert ex.__getattribute__(field.name)[1][0] == expected_data_dict[
                data_key]


@pytest.mark.parametrize(
    "json_data, fields_dict, exception_type",
    [
        (
                '{"text": "this is a review", "rating": 4.5, "sentiment": 1}',
                {"not_text": (MockField("words"), MockField("chars")),
                 "rating": MockField("label"),
                 "sentiment": (MockField("sentiment"), MockField("polarity"))},
                ValueError
        ),
    ]
)
def test_fromJSON_exception(json_data, fields_dict, exception_type):
    try:
        Example.fromJSON(json_data, fields_dict)
    except Exception as e:
        t = type(e)

    assert t == exception_type


@pytest.mark.parametrize(
    "data_list, fields_list",
    [
        (
                ["data"],
                [(
                        MockField("docs"), MockField("paragraphs"),
                        MockField("sents"),
                        MockField("words"),
                        MockField("syllables"), MockField("chars"))]
        ),
        (
                ["this is a review", 4.5, 1],
                [(MockField("words"), MockField("chars")), MockField("label"),
                 (MockField("sentiment"), MockField("polarity"))]
        ),
    ]
)
def test_fromlist(data_list, fields_list):
    ex = Example.fromlist(data_list, fields_list)

    for data, fields in zip(data_list, fields_list):
        if not isinstance(fields, tuple):
            fields = (fields,)

        for field in fields:
            assert ex.__getattribute__(field.name)[0] == data
            assert ex.__getattribute__(field.name)[1][0] == data


@pytest.mark.parametrize(
    "csv_line, fields_dict, field_to_index, delimiter, expected_data_dict",
    [
        (
                ",".join(["this is a review", "4.5", "1"]),
                {"text": (MockField("words"), MockField("chars")),
                 "rating": MockField("label"),
                 "sentiment": (MockField("sentiment"), MockField("polarity"))},
                {"text": 0, "rating": 1, "sentiment": 2},
                ",",
                {"text": "this is a review", "rating": "4.5", "sentiment": "1"}
        ),
        (
                "||".join(["this is a review", "bla"]),
                {
                    "text": (MockField("words"), MockField("chars")),
                    "sentiment": (
                            MockField("sentiment"), MockField("polarity")
                    )
                },
                {"text": 0, "sentiment": 1},
                "||",
                {"text": "this is a review", "sentiment": "bla"}
        )
    ]
)
def test_fromCSV_fields_is_dict(csv_line, fields_dict, field_to_index,
                                delimiter, expected_data_dict):
    ex = Example.fromCSV(csv_line, fields_dict, field_to_index, delimiter)

    for data_key, data_val in expected_data_dict.items():
        fields = fields_dict[data_key]

        if not isinstance(fields, tuple):
            fields = (fields,)

        for field in fields:
            assert ex.__getattribute__(field.name)[0] == expected_data_dict[
                data_key]
            assert ex.__getattribute__(field.name)[1][0] == expected_data_dict[
                data_key]


@pytest.mark.parametrize(
    "csv_line, fields_list, delimiter",
    [
        (
                "blabla",
                [(
                        MockField("docs"), MockField("paragraphs"),
                        MockField("sents"),
                        MockField("words"),
                        MockField("syllables"), MockField("chars"))],
                ","
        ),
        (
                ",".join(["this is a review", "4.5", "1"]),
                [(MockField("words"), MockField("chars")), MockField("label"),
                 (MockField("sentiment"), MockField("polarity"))],
                ","
        ),
        (
                "\t".join(["this is a review", "4.5", "1"]),
                [(MockField("words"), MockField("chars")), MockField("label"),
                 (MockField("sentiment"), MockField("polarity"))],
                "\t"
        )
    ]
)
def test_fromCSV_fields_is_list(csv_line, fields_list, delimiter):
    ex = Example.fromCSV(csv_line, fields_list, None, delimiter)

    elements = map(lambda s: s.strip(), csv_line.split(delimiter))
    for data, fields in zip(elements, fields_list):
        if not isinstance(fields, tuple):
            fields = (fields,)

        for field in fields:
            assert ex.__getattribute__(field.name)[0] == data
            assert ex.__getattribute__(field.name)[1][0] == data


@pytest.mark.parametrize(
    "data, fields_list, expected_attributes",
    [
        (
                "(S (NP I) (VP (V saw) (NP him)))",
                [(MockField("text"), MockField("chars")), MockField("label")],
                {"text": "I saw him", "chars": "I saw him", "label": "S"}
        )
    ]
)
def test_fromtree_no_subtrees(data, fields_list, expected_attributes):
    ex = Example.fromtree(data, fields_list, subtrees=False)
    for fields in fields_list:
        if not isinstance(fields, tuple):
            fields = (fields,)

        for f in fields:
            assert ex.__getattribute__(f.name)[0] == expected_attributes[
                f.name]
            assert ex.__getattribute__(f.name)[1][0] == expected_attributes[
                f.name]


@pytest.mark.parametrize(
    "data, fields_list, expected_attributes_list",
    [
        (
                "(S (NP I) (VP (V saw) (NP him)))",
                [(MockField("text"), MockField("chars")), MockField("label")],
                [
                    {"text": "I saw him", "chars": "I saw him", "label": "S"},
                    {"text": "I", "chars": "I", "label": "NP"},
                    {"text": "saw him", "chars": "saw him", "label": "VP"},
                    {"text": "saw", "chars": "saw", "label": "V"},
                    {"text": "him", "chars": "him", "label": "NP"},
                ]
        )
    ]
)
def test_fromtree_with_subtrees(data, fields_list, expected_attributes_list):
    examples = Example.fromtree(data, fields_list, subtrees=True)

    assert len(examples) == len(expected_attributes_list)
    for ex, expected_attributes in zip(examples, expected_attributes_list):
        for fields in fields_list:
            if not isinstance(fields, tuple):
                fields = (fields,)

            for f in fields:
                assert ex.__getattribute__(f.name)[0] == expected_attributes[
                    f.name]
                assert ex.__getattribute__(f.name)[1][0] == \
                    expected_attributes[f.name]
