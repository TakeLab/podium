import importlib
import os

import pandas as pd
import pytest

from podium.datasets.tabular_dataset import TabularDataset
from podium.field import Field, LabelField
from podium.utils.general_utils import load_spacy_model_or_raise
from podium.vocab import Vocab


def pytest_configure(config):
    # turn warnings into errors
    config.addinivalue_line("filterwarnings", "error")

    # ignore some warnings
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")
    config.addinivalue_line("filterwarnings", "ignore::PendingDeprecationWarning")
    config.addinivalue_line("filterwarnings", "ignore::ImportWarning")
    config.addinivalue_line("filterwarnings", "ignore::ResourceWarning")

    # define additional markers
    config.addinivalue_line(
        "markers",
        "require_package(package): mark test to run only if the required package is installed",
    )
    config.addinivalue_line(
        "markers",
        "require_spacy_model(model): mark test to run only if the required SpaCy model is installed",
    )


def pytest_runtest_setup(item):
    required_packages = [
        mark.args[0] for mark in item.iter_markers(name="require_package")
    ]
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"test requires the {package} package")

    models = [mark.args[0] for mark in item.iter_markers(name="require_spacy_model")]
    try:
        for model in models:
            load_spacy_model_or_raise(model)
    except OSError:
        pytest.skip(f"test requires the {model} SpaCy model installed")


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def vocab(tabular_dataset_fields):
    return tabular_dataset_fields["text"].vocab


@pytest.fixture
@pytest.mark.usefixtures("json_file_path")
def cache_disabled_tabular_dataset(json_file_path):
    return create_tabular_dataset_from_json(
        tabular_dataset_fields(disable_numericalize_caching=True), json_file_path
    )


@pytest.fixture
@pytest.mark.usefixtures("json_file_path")
def length_included_tabular_dataset(json_file_path):
    td = create_tabular_dataset_from_json(
        tabular_dataset_fields(include_lengths=True), json_file_path
    )
    return td


@pytest.fixture
@pytest.mark.usefixtures("json_file_path")
def tabular_dataset(json_file_path):
    return create_tabular_dataset_from_json(tabular_dataset_fields(), json_file_path)


def tabular_dataset_fields(
    fixed_length=None, disable_numericalize_caching=False, include_lengths=False
):
    text = Field(
        "text",
        numericalizer=Vocab(eager=True),
        fixed_length=fixed_length,
        allow_missing_data=False,
        include_lengths=include_lengths,
        disable_numericalize_caching=disable_numericalize_caching,
    )
    text_missing = Field(
        "text_with_missing_data",
        numericalizer=Vocab(eager=True),
        fixed_length=fixed_length,
        allow_missing_data=True,
    )
    rating = LabelField("rating", numericalizer=float)

    fields = {"text": text, "text_with_missing_data": text_missing, "rating": rating}

    return fields


@pytest.fixture(name="tabular_dataset_fields")
def tabular_dataset_fields_fixture():
    return tabular_dataset_fields()


TABULAR_TEXT_WITH_MISSING = ("a b c", "a", "a b c d", None, "d b", "d c g", "b b b b b b")

TABULAR_TEXT = ("a b c", "a", "a b c d", "a", "d b", "d c g", "b b b b b b")

TABULAR_RATINGS = (2.5, 3.2, 1.1, 2.1, 5.4, 2.8, 1.9)


def tabular_data():
    return {
        "text": TABULAR_TEXT,
        "text_with_missing_data": TABULAR_TEXT_WITH_MISSING,
        "rating": TABULAR_RATINGS,
    }


@pytest.fixture(name="tabular_data")
def tabular_data_fixture():
    return tabular_data()


@pytest.mark.usefixtures("json_file_path")
def create_tabular_dataset_from_json(fields, json_file_path):
    return TabularDataset(json_file_path, fields, format="json", skip_header=False)
