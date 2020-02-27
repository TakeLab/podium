import os

import pandas as pd
import pytest

from takepod.datasets.tabular_dataset import TabularDataset
from takepod.storage.field import Field
from takepod.storage.vocab import Vocab


@pytest.fixture()
@pytest.mark.usefixtures("tabular_data")
def file_path(tmpdir, file_format, tabular_data):
    # tmpdir is a default pytest fixture
    path = os.path.join(tmpdir, "sample." + file_format)

    if file_format == "csv":
        create_temp_csv(path, ",", tabular_data)
    elif file_format == "tsv":
        create_temp_csv(path, "\t", tabular_data)
    else:
        create_temp_json(path, tabular_data)

    yield path


@pytest.fixture()
def json_file_path(tmpdir):
    # tmpdir is a default pytest fixture
    path = os.path.join(tmpdir, "sample.json")
    create_temp_json(path, tabular_data())

    yield path


def create_temp_json(path, data):
    df = pd.DataFrame(data)
    lines = (df.loc[i].to_json() + "\n" for i in df.index)

    with open(path, "w") as f:
        f.writelines(lines)


def create_temp_csv(path, delimiter, data):
    df = pd.DataFrame(data)
    df.to_csv(path, sep=delimiter, index=False)


@pytest.fixture()
def vocab(tabular_dataset_fields):
    return tabular_dataset_fields["text"].vocab


@pytest.fixture()
@pytest.mark.usefixtures("json_file_path")
def tabular_dataset(json_file_path):
    return create_tabular_dataset_from_json(tabular_dataset_fields(),
                                            json_file_path)


@pytest.fixture()
def tabular_dataset_fields(fixed_length=None):
    text = Field('text', eager=True, vocab=Vocab(),
                 fixed_length=fixed_length, allow_missing_data=False)
    text_missing = Field('text_with_missing_data', eager=True, vocab=Vocab(),
                         fixed_length=fixed_length, allow_missing_data=True)
    rating = Field('rating', tokenize=False, eager=False, is_target=True,
                   custom_numericalize=float)

    fields = {"text": text, "text_with_missing_data": text_missing, "rating": rating}

    return fields


TABULAR_TEXT_WITH_MISSING = (
    "a b c",
    "a",
    "a b c d",
    None,
    "d b",
    "d c g",
    "b b b b b b"
)

TABULAR_TEXT = (
    "a b c",
    "a",
    "a b c d",
    "a",
    "d b",
    "d c g",
    "b b b b b b"
)

TABULAR_RATINGS = (2.5, 3.2, 1.1, 2.1, 5.4, 2.8, 1.9)


@pytest.fixture()
def tabular_data():
    return {
        "text": TABULAR_TEXT,
        "text_with_missing_data": TABULAR_TEXT_WITH_MISSING,
        "rating": TABULAR_RATINGS,
    }


@pytest.mark.usefixtures("json_file_path")
def create_tabular_dataset_from_json(fields, json_file_path):
    return TabularDataset(json_file_path, "json", fields, skip_header=False)
