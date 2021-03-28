import csv
import os

from podium.datasets.dataset import Dataset

from .example_factory import ExampleFactory


class TabularDataset(Dataset):
    """
    A dataset type for data stored in a single CSV, TSV or JSON file, where each
    row of the file is a single example.
    """

    def __init__(
        self,
        path,
        fields,
        format="csv",
        line2example=None,
        skip_header=False,
        csv_reader_params={},
        **kwargs,
    ):
        """
        Creates a TabularDataset from a file containing the data rows and an
        object containing all the fields that we are interested in.

        Parameters
        ----------
        path : str
            Path to the data file.
        fields : (list | dict)
            A mapping from data columns to example fields.
            This allows the user to rename columns from the data file,
            to create multiple fields from the same column and also to
            select only a subset of columns to load.

            A value stored in the list/dict can be either a Field
            (1-to-1 mapping), a tuple of Fields (1-to-n mapping) or
            None (ignore column).

            If type is list, then it should map from the column index to
            the corresponding field/s (i.e. the fields in the list should
            be in the same order as the columns in the file). Also, the
            format must be CSV or TSV.

            If type is dict, then it should be a map from the column name
            to the corresponding field/s. Column names not present in
            the dict's keys are ignored. If the format is CSV/TSV,
            then the data file must have a header
            (column names need to be known).
        format : str
            The format of the data file. Has to be either "CSV", "TSV",
            "JSON" (case-insensitive). Ignored if `line2example` is
            not None. Defaults to "CSV".
        line2example : callable
            The function mapping from a file line to Fields.
            In case your dataset is not in one of the standardized formats,
            you can provide a function which performs a custom split for
            each input line.
        skip_header : bool
            Whether to skip the first line of the input file.
            If format is CSV/TSV and 'fields' is a dict, then skip_header
            must be False and the data file must have a header.
            Default is False.
        delimiter: str
            Delimiter used to separate columns in a row.
            If set to None, the default delimiter for the given format will
            be used.
        csv_reader_params : dict
            Parameters to pass to the csv reader. Only relevant when
            format is csv or tsv.
            See https://docs.python.org/3/library/csv.html#csv.reader
            for more details.

        Raises
        ------
        ValueError
            If the format given is not one of "CSV", "TSV" or "JSON"
            and line2example is not set.
            If fields given as a dict and skip_header is True.
            If format is "JSON" and skip_header is True.
        """

        format = format.lower()

        with open(os.path.expanduser(path), encoding="utf8") as f:

            # Skip header prior to custom line2example in case
            # the header is in a different format so we don't
            # cause an error.
            if skip_header:
                if format == "json":
                    raise ValueError(
                        f"When using a {format} file, skip_header \
                                       must be False."
                    )
                elif format in {"csv", "tsv", "custom"} and isinstance(fields, dict):
                    raise ValueError(
                        f"When using a dict to specify fields with a {format} "
                        "file, skip_header must be False and the file must "
                        "have a header."
                    )

                # skip the header
                next(f)

            if line2example is not None:
                reader = (line2example(line) for line in f)
                format = "custom"
            elif format in {"csv", "tsv"}:
                delimiter = "," if format == "csv" else "\t"
                reader = csv.reader(f, delimiter=delimiter, **csv_reader_params)
            elif format == "json":
                reader = f
            else:
                raise ValueError(f"Invalid format: {format}")

            # create a list of examples
            examples = create_examples(reader, format, fields)

        # create a Dataset with lists of examples and fields
        super(TabularDataset, self).__init__(examples, fields, **kwargs)


def create_examples(reader, format, fields):
    """
    Creates a list of examples from the given line reader and fields (see
    TabularDataset.__init__ docs for more info on the fields).

    Parameters
    ----------
    reader
        A reader object that reads one line at a time. Yields either strings
        (when format is JSON) or lists of values (when format is CSV/TSV).
    format : str
        Format of the data file that is being read. Can be either CSV,
        TSV or JSON.
    fields : (list | dict)
        A list or dict of fields (see TabularDataset.__init__ docs for more
        info).

    Returns
    -------
    list
        A list of created examples.

    Raises
    ------
    ValueError
        If format is JSON and skip_header is True.
        If format is CSV/TSV, the fields are given as a dict and
        skip_header is True.
    """

    # if format is CSV/TSV and fields is a dict, transform it to a list
    if format in {"csv", "tsv", "custom"} and isinstance(fields, dict):
        # we need a header to know the column names
        header = next(reader)
        # columns not present in the fields dict are ignored (None)
        fields = [fields.get(column, None) for column in header]

    # fields argument is the same for all examples
    # from_list is used for CSV/TSV because csv_reader yields data rows as
    # lists, not strings
    example_factory = ExampleFactory(fields)
    make_example_function = {
        "json": example_factory.from_json,
        "csv": example_factory.from_list,
        "tsv": example_factory.from_list,
        "custom": example_factory.from_list,
    }

    make_example = make_example_function[format]

    # map each line from the reader to an example
    examples = map(make_example, reader)

    return list(examples)
