from typing import Dict, Iterator, List, Optional, Union

import pandas as pd

from podium.field import Field

from .example_factory import Example, ExampleFactory


FieldType = Optional[Union[Field, List[Field]]]


def pandas_to_examples(
    df: pd.DataFrame,
    fields: Optional[Union[Dict[str, Field], List[Field]]] = None,
    index_field=None,
) -> Iterator[Example]:

    if fields is None:
        fields = {}

    if isinstance(fields, (list, tuple)):
        field_list = list(fields)
        if len(field_list) != len(df.columns):
            raise ValueError(
                f"Invalid number of fields. "
                f"Number of fields is {len(field_list)}, "
                f"number of columns in dataframe is {len(df.columns)}."
            )

    elif isinstance(fields, dict):
        field_list = [None] * len(df.columns)
        column_name_index = {name: index for index, name in enumerate(df.columns)}
        for column_name, field in fields.items():
            if column_name not in column_name_index:
                raise KeyError(
                    f"Column '{column_name}' for field '{field}' "
                    f"not present in the dataframe."
                )

            field_list[column_name_index[column_name]] = field

    else:
        raise TypeError(
            f"Invalid 'fields' type. Must be either field, tuple or dict. "
            f"Passed type: '{type(fields).__name__}'"
        )

    field_list = [index_field] + field_list
    example_factory = ExampleFactory(field_list)

    for row in df.itertuples():
        yield example_factory.from_list(row)
