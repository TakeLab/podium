import pytest

from takepod.storage import ExampleFactory, Field

name_field = Field("Name",
                   store_as_raw=True)

score_field = Field("Score",
                    custom_numericalize=int,
                    tokenize=False)

favorite_food_field = Field("Favorite_food",
                            store_as_raw=True,
                            tokenize=False)

field_list = [name_field, score_field, favorite_food_field]

field_dict = {field.name: field for field in field_list}


@pytest.mark.parametrize('example_values',
                         [
                             ["Mark Dark", 5, "Hawaiian pizza"],
                             ["Stephen Smith", 10, "Fried squid"],
                             ["Ann Mann", 15, "Tofu"]
                         ]
                         )
def test_create_from_list(example_values):
    example_factory = ExampleFactory(field_list)
    example = example_factory.from_list(example_values)

    raw, tokenized = example.Name
    assert tokenized == example_values[0].split()

    raw, tokenized = example.Score
    assert raw == example_values[1]

    raw, tokenized = example.Favorite_food
    assert raw == example_values[2]


@pytest.mark.parametrize('example_values',
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
def test_create_from_dict(example_values):
    example_factory = ExampleFactory(field_dict)
    example = example_factory.from_dict(example_values)

    raw, tokenized = example.Name
    assert raw == example_values['Name']
    assert tokenized == example_values['Name'].split()

    raw, tokenized = example.Score
    assert raw == example_values['Score']

    raw, tokenized = example.Favorite_food
    assert raw == example_values['Favorite_food']


@pytest.mark.parametrize('example_values, example_xml_string',
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
def test_create_from_xml_string(example_values, example_xml_string):
    example_factory = ExampleFactory(field_dict)
    example = example_factory.from_xml_str(example_xml_string)

    raw, tokenized = example.Name
    assert raw == example_values['Name']
    assert tokenized == example_values['Name'].split()

    raw, tokenized = example.Score
    assert raw == str(example_values['Score'])

    raw, tokenized = example.Favorite_food
    assert raw == example_values['Favorite_food']


@pytest.mark.parametrize('example_values, example_json_string',
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
def test_create_from_json_string(example_values, example_json_string):
    example_factory = ExampleFactory(field_dict)
    example = example_factory.from_json(example_json_string)

    raw, tokenized = example.Name
    assert raw == example_values['Name']
    assert tokenized == example_values['Name'].split()

    raw, tokenized = example.Score
    assert raw == example_values['Score']

    raw, tokenized = example.Favorite_food
    assert raw == example_values['Favorite_food']


@pytest.mark.parametrize('example_values, example_csv_string',
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
def test_create_from_csv(example_values, example_csv_string):
    example_factory = ExampleFactory(field_list)
    example = example_factory.from_csv(example_csv_string)

    raw, tokenized = example.Name
    assert raw == example_values[0]
    assert tokenized == example_values[0].split()

    raw, tokenized = example.Score
    assert int(raw) == example_values[1]

    raw, tokenized = example.Favorite_food
    assert raw == example_values[2]


@pytest.mark.parametrize('example_values',
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
def test_multiple_output_for_input_dict(example_values):
    lower_case_name_field = Field("Lowercase_name")
    lower_case_name_field.add_pretokenize_hook(str.lower)

    upper_case_name_field = Field("Uppercase_name")
    upper_case_name_field.add_pretokenize_hook(str.upper)

    test_field_dict = dict(field_dict)
    test_field_dict["Name"] = (field_dict['Name'],
                               lower_case_name_field,
                               upper_case_name_field)

    example_factory = ExampleFactory(test_field_dict)
    example = example_factory.from_dict(example_values)

    raw, tokenized = example.Name
    assert raw == example_values['Name']
    assert tokenized == example_values['Name'].split()

    raw, tokenized = example.Lowercase_name
    assert raw == example_values['Name'].lower()
    assert tokenized == example_values['Name'].lower().split()

    raw, tokenized = example.Uppercase_name
    assert raw == example_values['Name'].upper()
    assert tokenized == example_values['Name'].upper().split()

    raw, tokenized = example.Score
    assert raw == example_values['Score']

    raw, tokenized = example.Favorite_food
    assert raw == example_values['Favorite_food']


@pytest.mark.parametrize('example_values',
                         [
                             ["Mark Dark", 5, "Hawaiian pizza"],
                             ["Stephen Smith", 10, "Fried squid"],
                             ["Ann Mann", 15, "Tofu"]
                         ]
                         )
def test_multiple_output_for_input_list(example_values):
    lower_case_name_field = Field("Lowercase_name")
    lower_case_name_field.add_pretokenize_hook(str.lower)

    upper_case_name_field = Field("Uppercase_name")
    upper_case_name_field.add_pretokenize_hook(str.upper)

    test_field_list = list(field_list)

    test_field_list[0] = (test_field_list[0],
                          lower_case_name_field,
                          upper_case_name_field)

    example_factory = ExampleFactory(test_field_list)
    example = example_factory.from_list(example_values)

    raw, tokenized = example.Name
    assert raw == example_values[0]
    assert tokenized == example_values[0].split()

    raw, tokenized = example.Lowercase_name
    assert raw == example_values[0].lower()
    assert tokenized == example_values[0].lower().split()

    raw, tokenized = example.Uppercase_name
    assert raw == example_values[0].upper()
    assert tokenized == example_values[0].upper().split()

    raw, tokenized = example.Score
    assert raw == example_values[1]

    raw, tokenized = example.Favorite_food
    assert raw == example_values[2]
