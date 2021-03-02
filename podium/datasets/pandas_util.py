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
    """
    Utility function for lazy loading of Examples from pandas Dataframes
    Parameters
    ----------
    df: pandas.Dataframe
        Pandas dataframe from which data will be taken.

    fields: Optional[Union[Dict[str, Field], List[Field]]]
        A mapping from dataframe columns to example fields.
            This allows the user to rename columns from the data file,
            to create multiple fields from the same column and also to
            select only a subset of columns to load.

            A value stored in the list/dict can be either a Field
            (1-to-1 mapping), a tuple of Fields (1-to-n mapping) or
            None (ignore column).

            If type is list, then it should map from the column index to
            the corresponding field/s (i.e. the fields in the list should
            be in the same order as the columns in the dataframe).

            If type is dict, then it should be a map from the column name
            to the corresponding field/s. Column names not present in
            the dict's keys are ignored.

    index_field: Optional[Field]
            Field which will be used to process the index column of the Dataframe.
            If None, the index column will be ignored.

    Returns
    -------
    Iterator[Example]
        Iterator iterating over Examples created from the columns of the passed Dataframe.

    """
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
