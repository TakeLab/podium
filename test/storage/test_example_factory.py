import pytest

from takepod.storage import ExampleFactory, Field, ExampleFormat

name_field = Field("Name",
                   store_as_raw=True,
                   tokenizer="split")

score_field = Field("Score",
                    custom_numericalize=int,
                    store_as_raw=True,
                    tokenize=False)

favorite_food_field = Field("Favorite_food",
                            store_as_raw=True,
                            tokenize=False)

field_list = [name_field, score_field, favorite_food_field]

field_dict = {field.name: field for field in field_list}


@pytest.mark.parametrize('expected_values',
                         [
                             ["Mark Dark", 5, "Hawaiian pizza"],
                             ["Stephen Smith", 10, "Fried squid"],
                             ["Ann Mann", 15, "Tofu"]
                         ]
                         )
def test_create_from_list(expected_values):
    example_factory = ExampleFactory(field_list)
    example = example_factory.from_list(expected_values)

    raw, tokenized = example.Name
    assert tokenized == expected_values[0].split()

    raw, tokenized = example.Score
    assert raw == expected_values[1]

    raw, tokenized = example.Favorite_food
    assert raw == expected_values[2]


@pytest.mark.parametrize('expected_values',
                         [
                             {"Name": "Mark Dark",
                              "Score": 5,
                              "Favorite_food": "Hawaiian pizza"},
                             {"Name": "Stephen Smith", "Score": 10,
                              "Favorite_food": "Fried squid"},
                             {"Name": "Ann Mann",
                              "Score": 15,
                              "Favorite_food": "Tofu"}
                         ]
                         )
def test_create_from_dict(expected_values):
    example_factory = ExampleFactory(field_dict)
    example = example_factory.from_dict(expected_values)

    raw, tokenized = example.Name
    assert raw == expected_values['Name']
    assert tokenized == expected_values['Name'].split()

    raw, tokenized = example.Score
    assert raw == expected_values['Score']

    raw, tokenized = example.Favorite_food
    assert raw == expected_values['Favorite_food']


@pytest.mark.parametrize('expected_values, example_xml_string',
                         [
                             ({"Name": "Mark Dark",
                               "Score": 5,
                               "Favorite_food": "Hawaiian pizza"
                               },
                              "<Example>"
                              " <Name>Mark Dark</Name>\n"
                              " <Score>5</Score>\n"
                              " <Favorite_food>Hawaiian pizza</Favorite_food>\n"
                              "</Example>"
                              ),

                             ({"Name": "Stephen Smith",
                               "Score": 10,
                               "Favorite_food": "Fried squid"
                               },
                              "<Example>"
                              " <Name>Stephen Smith</Name>\n"
                              " <Score>10</Score>\n"
                              " <Favorite_food>Fried squid</Favorite_food>\n"
                              "</Example>"
                              ),
                             ({"Name": "Ann Mann",
                               "Score": 15,
                               "Favorite_food": "Tofu"
                               },
                              "<Example>"
                              " <Name>Ann Mann</Name>\n"
                              " <Score>15</Score>\n"
                              " <Favorite_food>Tofu</Favorite_food>\n"
                              "</Example>"

                              )
                         ]
                         )
def test_create_from_xml_string(expected_values, example_xml_string):
    example_factory = ExampleFactory(field_dict)
    example = example_factory.from_xml_str(example_xml_string)

    raw, tokenized = example.Name
    assert raw == expected_values['Name']
    assert tokenized == expected_values['Name'].split()

    raw, tokenized = example.Score
    assert raw == str(expected_values['Score'])

    raw, tokenized = example.Favorite_food
    assert raw == expected_values['Favorite_food']


@pytest.mark.parametrize('expected_values, example_json_string',
                         [
                             ({"Name": "Mark Dark",
                               "Score": 5,
                               "Favorite_food": "Hawaiian pizza"},
                              "{"
                              " \"Name\": \"Mark Dark\",\n"
                              " \"Score\":5,\n"
                              " \"Favorite_food\": \"Hawaiian pizza\""
                              "}"),

                             ({"Name": "Stephen Smith",
                               "Score": 10,
                               "Favorite_food": "Fried squid"
                               },
                              "{"
                              " \"Name\": \"Stephen Smith\",\n"
                              " \"Score\":10,\n"
                              " \"Favorite_food\": \"Fried squid\""
                              "}"),
                             ({"Name": "Ann Mann",
                               "Score": 15,
                               "Favorite_food": "Tofu"
                               },
                              "{"
                              " \"Name\": \"Ann Mann\",\n"
                              " \"Score\":15,\n"
                              " \"Favorite_food\": \"Tofu\""
                              "}"

                              )
                         ]
                         )
def test_create_from_json_string(expected_values, example_json_string):
    example_factory = ExampleFactory(field_dict)
    example = example_factory.from_json(example_json_string)

    raw, tokenized = example.Name
    assert raw == expected_values['Name']
    assert tokenized == expected_values['Name'].split()

    raw, tokenized = example.Score
    assert raw == expected_values['Score']

    raw, tokenized = example.Favorite_food
    assert raw == expected_values['Favorite_food']


@pytest.mark.parametrize('expected_values, example_csv_string',
                         [
                             (["Mark Dark", 5, "Hawaiian pizza"],
                              "Mark Dark,5,Hawaiian pizza"
                              ),
                             (["Stephen Smith", 10, "Fried squid"],
                              "Stephen Smith,10,Fried squid"
                              ),
                             (["Ann Mann", 15, "Tofu"],
                              "Ann Mann,15,Tofu"
                              )
                         ]
                         )
def test_create_from_csv(expected_values, example_csv_string):
    example_factory = ExampleFactory(field_list)
    example = example_factory.from_csv(example_csv_string)

    raw, tokenized = example.Name
    assert raw == expected_values[0]
    assert tokenized == expected_values[0].split()

    raw, tokenized = example.Score
    assert int(raw) == expected_values[1]

    raw, tokenized = example.Favorite_food
    assert raw == expected_values[2]


@pytest.mark.parametrize('expected_values, example_tsv_string',
                         [
                             (["Mark Dark", 5, "Hawaiian pizza"],
                              "Mark Dark\t5\tHawaiian pizza"
                              ),
                             (["Stephen Smith", 10, "Fried squid"],
                              "Stephen Smith\t10\tFried squid"
                              ),
                             (["Ann Mann", 15, "Tofu"],
                              "Ann Mann\t15\tTofu"
                              )
                         ]
                         )
def test_create_from_tsv(expected_values, example_tsv_string):
    example_factory = ExampleFactory(field_list)
    example = example_factory.from_csv(example_tsv_string, delimiter="\t")

    raw, tokenized = example.Name
    assert raw == expected_values[0]
    assert tokenized == expected_values[0].split()

    raw, tokenized = example.Score
    assert int(raw) == expected_values[1]

    raw, tokenized = example.Favorite_food
    assert raw == expected_values[2]


@pytest.mark.parametrize('expected_values',
                         [
                             {"Name": "Mark Dark",
                              "Score": 5,
                              "Favorite_food": "Hawaiian pizza"},
                             {"Name": "Stephen Smith",
                              "Score": 10,
                              "Favorite_food": "Fried squid"
                              },
                             {"Name": "Ann Mann",
                              "Score": 15,
                              "Favorite_food": "Tofu"
                              }
                         ]
                         )
def test_multiple_output_for_input_dict(expected_values):
    lower_case_name_field = Field("Lowercase_name", store_as_raw=True)
    lower_case_name_field.add_pretokenize_hook(str.lower)

    upper_case_name_field = Field("Uppercase_name", store_as_raw=True)
    upper_case_name_field.add_pretokenize_hook(str.upper)

    test_field_dict = dict(field_dict)
    test_field_dict["Name"] = (field_dict['Name'],
                               lower_case_name_field,
                               upper_case_name_field)

    example_factory = ExampleFactory(test_field_dict)
    example = example_factory.from_dict(expected_values)

    raw, tokenized = example.Name
    assert raw == expected_values['Name']
    assert tokenized == expected_values['Name'].split()

    raw, tokenized = example.Lowercase_name
    assert raw == expected_values['Name'].lower()
    assert tokenized == expected_values['Name'].lower().split()

    raw, tokenized = example.Uppercase_name
    assert raw == expected_values['Name'].upper()
    assert tokenized == expected_values['Name'].upper().split()

    raw, tokenized = example.Score
    assert raw == expected_values['Score']

    raw, tokenized = example.Favorite_food
    assert raw == expected_values['Favorite_food']


@pytest.mark.parametrize('expected_values',
                         [
                             ["Mark Dark", 5, "Hawaiian pizza"],
                             ["Stephen Smith", 10, "Fried squid"],
                             ["Ann Mann", 15, "Tofu"]
                         ]
                         )
def test_multiple_output_for_input_list(expected_values):
    lower_case_name_field = Field("Lowercase_name", store_as_raw=True)
    lower_case_name_field.add_pretokenize_hook(str.lower)

    upper_case_name_field = Field("Uppercase_name", store_as_raw=True)
    upper_case_name_field.add_pretokenize_hook(str.upper)

    test_field_list = list(field_list)

    test_field_list[0] = (test_field_list[0],
                          lower_case_name_field,
                          upper_case_name_field)

    example_factory = ExampleFactory(test_field_list)
    example = example_factory.from_list(expected_values)

    raw, tokenized = example.Name
    assert raw == expected_values[0]
    assert tokenized == expected_values[0].split()

    raw, tokenized = example.Lowercase_name
    assert raw == expected_values[0].lower()
    assert tokenized == expected_values[0].lower().split()

    raw, tokenized = example.Uppercase_name
    assert raw == expected_values[0].upper()
    assert tokenized == expected_values[0].upper().split()

    raw, tokenized = example.Score
    assert raw == expected_values[1]

    raw, tokenized = example.Favorite_food
    assert raw == expected_values[2]


@pytest.mark.parametrize('expected_values',
                         [
                             ["Mark Dark", 5, "Hawaiian pizza"],
                             ["Stephen Smith", 10, "Fried squid"],
                             ["Ann Mann", 15, "Tofu"]
                         ]
                         )
def test_ignore_values_list(expected_values):
    fields = [None, None, favorite_food_field]
    example_factory = ExampleFactory(fields)
    example = example_factory.from_list(expected_values)

    # one is original field and one is for cached value
    assert hasattr(example, "Favorite_food")
    assert hasattr(example, "Favorite_food_")

    raw, _ = example.Favorite_food
    assert raw == expected_values[2]


@pytest.mark.parametrize('expected_values',
                         [
                             {"Name": "Mark Dark",
                              "Score": 5,
                              "Favorite_food": "Hawaiian pizza"},
                             {"Name": "Stephen Smith", "Score": 10,
                              "Favorite_food": "Fried squid"},
                             {"Name": "Ann Mann",
                              "Score": 15,
                              "Favorite_food": "Tofu"}
                         ]
                         )
def test_ignore_values_dict(expected_values):
    fields = {'Name': name_field}
    example_factory = ExampleFactory(fields)
    example = example_factory.from_dict(expected_values)

    # one is original field and one is for cached value
    assert hasattr(example, "Name")
    assert hasattr(example, "Name_")

    raw, _ = example.Name
    assert raw == expected_values['Name']


@pytest.mark.parametrize('expected_values',
                         [
                             ["Mark Dark", 5, "Hawaiian pizza"],
                             ["Stephen Smith", 10, "Fried squid"],
                             ["Ann Mann", 15, "Tofu"]
                         ]
                         )
def test_cache_data_field_from_list(expected_values):
    example_factory = ExampleFactory(field_list)
    example = example_factory.from_list(expected_values)

    for field in field_list:
        field_name = field.name

        assert hasattr(example, field_name)
        assert hasattr(example, "{}_".format(field_name))


@pytest.mark.parametrize('expected_values',
                         [
                             {"Name": "Mark Dark",
                              "Score": 5,
                              "Favorite_food": "Hawaiian pizza"},
                             {"Name": "Stephen Smith", "Score": 10,
                              "Favorite_food": "Fried squid"},
                             {"Name": "Ann Mann",
                              "Score": 15,
                              "Favorite_food": "Tofu"}
                         ]
                         )
def test_cache_data_field_from_dict(expected_values):
    example_factory = ExampleFactory(field_dict)
    example = example_factory.from_dict(expected_values)

    for field in field_dict.values():
        field_name = field.name

        assert hasattr(example, field_name)
        assert hasattr(example, "{}_".format(field_name))


def test_from_format():
    list_example_factory = ExampleFactory(field_list)

    list_data = ["Mark Dark", 5, "Hawaiian pizza"]
    example = list_example_factory.from_format(list_data, ExampleFormat.LIST)

    assert example.Name[0] == list_data[0]
    assert example.Score[0] == list_data[1]
    assert example.Favorite_food[0] == list_data[2]

    dict_example_factory = ExampleFactory(field_dict)
    dict_data = {"Name": "Mark Dark",
                 "Score": 5,
                 "Favorite_food": "Hawaiian pizza"}

    example = dict_example_factory.from_format(dict_data, ExampleFormat.DICT)
    assert example.Name[0] == dict_data["Name"]
    assert example.Score[0] == dict_data["Score"]
    assert example.Favorite_food[0] == dict_data["Favorite_food"]
    # TODO extend testing to other formats?
