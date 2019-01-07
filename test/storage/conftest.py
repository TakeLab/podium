import os

import pandas as pd
import pytest


@pytest.fixture()
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


def create_temp_json(path, data):
    df = pd.DataFrame(data)
    lines = (df.loc[i].to_json() + "\n" for i in df.index)

    with open(path, "w") as f:
        f.writelines(lines)


def create_temp_csv(path, delimiter, data):
    df = pd.DataFrame(data)
    df.to_csv(path, sep=delimiter, index=False)
