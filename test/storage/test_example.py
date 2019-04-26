import os
import dill
from xml.etree.ElementTree import ParseError
import pytest
from takepod.storage.example import Example


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
        ), (
            {"text": "this is a review", "rating": 4.5, "sentiment": 1,
             "source": "www.source.hr"},
            {"text": (MockField("words"), MockField("chars")),
             "rating": MockField("label"),
             "sentiment": (MockField("sentiment"), MockField("polarity")),
             "source": None
             }
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
    received_example = Example.from_dict(data_dict, fields_dict)

    field_data_tuples = ((fields_dict[k], v) for k, v in data_dict.items())
    expected_example = create_expected_example(field_data_tuples)

    assert received_example.__dict__ == expected_example.__dict__


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
def test_from_json_ok(json_data, fields_dict, expected_data_dict):
    received_example = Example.from_json(json_data, fields_dict)

    field_data_tuples = ((fields_dict[k], v) for k, v in
                         expected_data_dict.items())
    expected_example = create_expected_example(field_data_tuples)

    assert received_example.__dict__ == expected_example.__dict__


@pytest.mark.parametrize(
    "xml_str, fields_dict, expected_data_dict",
    [
        (
            "<example><text>this is a review</text><rating>4.5</rating>"
            "<sentiment>1</sentiment></example>",
            {"text": (MockField("words"), MockField("chars")),
             "rating": MockField("label"),
             "sentiment": (MockField("sentiment"), MockField("polarity"))},
            {"text": "this is a review", "rating": "4.5", "sentiment": "1"},
        ),
        (
            "<example><x>data</x></example>",
            {"x": (
                MockField("docs"), MockField("paragraphs"),
                MockField("sents"),
                MockField("words"),
                MockField("syllables"), MockField("chars"))},
            {"x": "data"},
        ),
        (
            "<x>data</x>",
            {"x": (
                MockField("docs"), MockField("paragraphs"),
                MockField("sents"),
                MockField("words"),
                MockField("syllables"), MockField("chars"))},
            {"x": "data"},
        ),
    ]
)
def test_fromxmlstr_ok(xml_str, fields_dict, expected_data_dict):
    received_example = Example.from_xml_str(data=xml_str,
                                            fields=fields_dict)

    field_data_tuples = ((fields_dict[k], v) for k, v in
                         expected_data_dict.items())
    expected_example = create_expected_example(field_data_tuples)

    assert received_example.__dict__ == expected_example.__dict__


@pytest.mark.parametrize(
    "xml_str, fields_dict",
    [
        (
            "<example><text>this is a review</text>"
            "<rating>4.5</rating><sentiment>1</sentiment></example>",
            {"not_text": (MockField("words"), MockField("chars")),
             "rating": MockField("label"),
             "sentiment": (MockField("sentiment"), MockField("polarity"))},
        ),
        (
            "<x>data</x>",
            {"y": (
                MockField("docs"), MockField("paragraphs"),
                MockField("sents"),
                MockField("words"),
                MockField("syllables"), MockField("chars"))},
        )
    ]
)
def test_fromxmlstr_value_error(xml_str, fields_dict):
    with pytest.raises(ValueError):
        Example.from_xml_str(xml_str, fields_dict)


@pytest.mark.parametrize(
    "xml_str, fields_dict",
    [
        (
            "<example><text>this is a review</text>"
            "<rating>4.5</rating><sentiment>1</sentiment>",
            {"not_text": (MockField("words"), MockField("chars")),
             "rating": MockField("label"),
             "sentiment": (MockField("sentiment"), MockField("polarity"))},
        ),
        (
            "<example><text>this is a review</text>"
            "<rating>4.5<sentiment>1</sentiment>",
            {"rating": MockField("label"),
             "sentiment": (MockField("sentiment"), MockField("polarity"))},
        ),
        (
            "data</x>",
            {"x": (
                MockField("docs"), MockField("paragraphs"),
                MockField("sents"),
                MockField("words"),
                MockField("syllables"), MockField("chars"))},
        )
    ]
)
def test_fromxmlstr_invalid_xml(xml_str, fields_dict):
    with pytest.raises(ParseError):
        Example.from_xml_str(xml_str, fields_dict)


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
        (
            ["this is a review", 4.5, 1, "this should be ignored"],
            [(MockField("words"), MockField("chars")), MockField("label"),
             (MockField("sentiment"), MockField("polarity")), None]
        ),
    ]
)
def test_fromlist(data_list, fields_list):
    received_example = Example.from_list(data_list, fields_list)
    expected_example = create_expected_example(zip(fields_list, data_list))

    assert received_example.__dict__ == expected_example.__dict__


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
            "|".join(["this is a review", "bla"]),
            {
                "text": (MockField("words"), MockField("chars")),
                "sentiment": (
                    MockField("sentiment"), MockField("polarity")
                )
            },
            {"text": 0, "sentiment": 1},
            "|",
            {"text": "this is a review", "sentiment": "bla"}
        )
    ]
)
def test_from_csv_fields_is_dict(csv_line, fields_dict, field_to_index,
                                 delimiter, expected_data_dict):
    received_example = Example.from_csv(csv_line, fields_dict, field_to_index,
                                        delimiter)

    field_data_tuples = ((fields_dict[k], v) for k, v in
                         expected_data_dict.items())
    expected_example = create_expected_example(field_data_tuples)

    assert received_example.__dict__ == expected_example.__dict__


@pytest.mark.parametrize(
    "csv_line, fields_dict, field_to_index, delimiter",
    [
        (
            ",".join(["this is a review", "4.5", "1"]),
            {"text": (MockField("words"), MockField("chars")),
             "rating": MockField("label"),
             "sentiment": (MockField("sentiment"), MockField("polarity"))},
            {"text": 0, "rating": 1, "sentiment": 2},
            ","
        ),
        (
            "|".join(["this is a review", "bla"]),
            {
                "text": (MockField("words"), MockField("chars")),
                "sentiment": (
                    MockField("sentiment"), MockField("polarity")
                )
            },
            {"text": 0, "sentiment": 1},
            "|"
        )
    ]
)
def test_from_csv_fields_pickle(csv_line, fields_dict, field_to_index,
                                delimiter, tmpdir):
    example = Example.from_csv(csv_line, fields_dict, field_to_index, delimiter)
    example_file = os.path.join(tmpdir, "example.pkl")

    with open(example_file, "wb") as fdata:
        dill.dump(example, fdata)

    with open(example_file, "rb") as fdata:
        loaded_example = dill.load(fdata)

        assert example.__dict__ == loaded_example.__dict__


@pytest.mark.parametrize(
    "csv_line, fields_list, delimiter, expected_data_list",
    [
        (
            "blabla",
            [(
                MockField("docs"), MockField("paragraphs"),
                MockField("sents"),
                MockField("words"),
                MockField("syllables"), MockField("chars")
            )],
            ",",
            ["blabla"]
        ),
        (
            ",".join(["this is a review", "4.5", "1"]),
            [(MockField("words"), MockField("chars")), MockField("label"),
             (MockField("sentiment"), MockField("polarity"))],
            ",",
            ["this is a review", "4.5", "1"]
        ),
        (
            ",".join(["\"this is, \"\"a\"\" review\"", "4.5", "1"]),
            [(MockField("words"), MockField("chars")), MockField("label"),
             (MockField("sentiment"), MockField("polarity"))],
            ",",
            ["this is, \"a\" review", "4.5", "1"]
        ),
        (
            "\t".join(["this is a review", "4.5", "1"]),
            [(MockField("words"), MockField("chars")), MockField("label"),
             (MockField("sentiment"), MockField("polarity"))],
            "\t",
            ["this is a review", "4.5", "1"]
        )
    ]
)
def test_from_csv_fields_is_list(csv_line, fields_list, delimiter,
                                 expected_data_list):
    received_example = Example.from_csv(csv_line, fields_list, None, delimiter)
    expected_example = create_expected_example(
        zip(fields_list, expected_data_list))

    assert received_example.__dict__ == expected_example.__dict__


@pytest.mark.parametrize(
    "data, fields_list, expected_attributes",
    [
        (
            "(S (NP I) (VP (V saw) (NP him)))",
            [(MockField("text"), MockField("chars")), MockField("label")],
            ["I saw him", "S"]
        )
    ]
)
def test_fromtree_no_subtrees(data, fields_list, expected_attributes):
    pytest.importorskip("nltk")
    received_example = Example.from_tree(data, fields_list, subtrees=False)
    expected_example = create_expected_example(
        zip(fields_list, expected_attributes))

    assert received_example.__dict__ == expected_example.__dict__


@pytest.mark.parametrize(
    "data, fields_list, expected_attributes_list",
    [
        (
            "(S (NP I) (VP (V saw) (NP him)))",
            [(MockField("text"), MockField("chars")), MockField("label")],
            [
                ["I saw him", "S"],
                ["I", "NP"],
                ["saw him", "VP"],
                ["saw", "V"],
                ["him", "NP"],
            ]
        )
    ]
)
def test_fromtree_with_subtrees(data, fields_list, expected_attributes_list):
    pytest.importorskip("nltk")
    received_examples = Example.from_tree(data, fields_list, subtrees=True)
    assert len(received_examples) == len(expected_attributes_list)

    for received_example, expected_attributes in zip(received_examples,
                                                     expected_attributes_list):
        expected_example = create_expected_example(
            zip(fields_list, expected_attributes))

        assert received_example.__dict__ == expected_example.__dict__


def create_expected_example(field_data_tuples):
    expected_example = Example()

    for fields, data_val in field_data_tuples:
        # the way MockField preprocesses data (raw, tokenized)
        expected_attribute = (data_val, [data_val])

        # None fields means that we ignore the corresponding column
        if fields is None:
            continue

        if not isinstance(fields, tuple):
            fields = (fields,)

        for field in fields:
            # set the column value to the field.name attribute of example
            setattr(expected_example, field.name, expected_attribute)

    return expected_example
