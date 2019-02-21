"""Module contains base classes for datasets."""
import csv
import io
import itertools
import os
import random
import json
from functools import partial
from abc import ABC
from collections import namedtuple

from takepod.storage.example import Example


class Dataset(ABC):
    """General purpose container for datasets defining some common methods.

     A dataset is a list of `Example` classes, along with the corresponding
    `Field` classes, which process the columns of each example.

    Attributes
    ----------
    examples : list
        A list of Example objects.
    fields : list
        A list of Field objects that were used to create examples.
    """

    def __init__(self, examples, fields):
        """Creates a dataset with the given examples and their fields.

        Parameters
        ----------
        examples : list
            A list of examples.
        fields : list
            A list of fields that the examples have been created with.
        """

        self.examples = examples
        self.fields = fields
        self.field_names = {field.name for field in fields}

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        """Yields values of the field with the given name for every
        example in the dataset.

        Parameters
        ----------
        attr : str
            The name of the field whose values are to be returned.

        Yields
        ------
            The field value of the next example.

        Raises
        ------
        AttributeError
            If there is no Field with the given name.
        """

        if attr in self.field_names:
            for x in self.examples:
                yield getattr(x, attr)
        else:
            raise AttributeError(f"Dataset has no field '{attr}'.")

    def finalize_fields(self, *args):
        """
        Builds vocabularies of all the non-eager fields in the dataset,
        from the Dataset objects given as *args and then finalizes all the
        fields.

        Parameters
        ----------
        args
            A variable number of Dataset objects from which to build the
            vocabularies for non-eager fields. If none provided, the
            vocabularies are built from this Dataset (self).
        """

        # if there are non-eager fields, we need to build their vocabularies
        fields_to_build = [f for f in self.fields if
                           not f.eager and f.use_vocab]
        if fields_to_build:
            # there can be multiple datasets we want to iterate over
            data_sources = list(
                filter(lambda arg: isinstance(arg, Dataset), args))

            # use self as a data source if no other given
            if not data_sources:
                data_sources.append(self)

            # for each example in each dataset,
            # update _all_ non-eager fields
            for dataset in data_sources:
                for example in dataset:
                    for field in fields_to_build:
                        field.update_vocab(*getattr(example, field.name))

        for field in self.fields:
            field.finalize()

    def split(self, split_ratio=0.7, stratified=False,
              strata_field_name=None,
              random_state=None, shuffle=True):
        """Creates train-(validation)-test splits from this dataset.

        The splits are new Dataset objects, each containing a part of this
        one's examples.

        Parameters
        ----------
            split_ratio : (float | list[float] | tuple[float])
                If type is float, a number in the interval (0.0, 1.0) denoting
                the amount of data to be used for the train split (the rest
                is used for test).
                If type is list or tuple, it should be of length 2 (or 3) and
                the numbers should denote the relative sizes of train, (valid)
                and test splits respectively.
                If the relative size for valid is missing (length is 2), only
                the train-test split is returned (valid is taken to be 0.0).
                Also, the relative sizes don't have to sum up to 1.0 (they are
                normalized automatically).
                The ratio must not be so unbalanced that it would result in
                either of the splits being empty (having zero elements).
                Default is 0.7 (for the train set).
            stratified : bool
                Whether the split should be stratified. A stratified split
                means that for each concrete value of the strata field, the
                given train-val-test ratio is preserved. Usually used on
                fields representing labels / classes, so that every class is
                present in each of our splits with the same percentage as in
                the entire dataset.
                Default is False.
            strata_field_name : str
                Name of the field that is to be used to do the stratified
                split. Only relevant when 'stratified' is true.
                If the name of the strata field is not provided (the default
                behaviour), the stratified split will be done over the first
                field that is a target (its 'is_target' attribute is True).
                Note that the values of the strata field have to be hashable.
                Default is None.
            random_state : int
                The random seed used for shuffling.

        Returns
        -------
        tuple[Dataset]
            Datasets for train, (validation) and test splits in that order,
            depending on the split ratios that were provided.

        Raises
        ------
        ValueError
            If the given split ratio is not in one of the valid forms.
            If the given split ratio is in a valid form, but wrong in the
            sense that it would result with at least one empty split.
            If stratified is True and the field with the given
            strata_field_name doesn't exist.
        """
        train_ratio, val_ratio, test_ratio = check_split_ratio(split_ratio)

        # For the permutations
        random.seed(random_state)

        if not stratified:
            train_data, val_data, test_data = rationed_split(
                self.examples,
                train_ratio,
                val_ratio,
                test_ratio,
                shuffle
            )
        else:
            strata_field_name = self._get_strata_field_name(strata_field_name)

            if strata_field_name is None:
                raise ValueError(
                    f"If strata_field_name is not provided, at least one"
                    f" field has to have is_target equal to True.")

            if strata_field_name not in self.field_names:
                raise ValueError(f"Invalid strata field name: "
                                 f"{strata_field_name}")

            train_data, val_data, test_data = stratified_split(
                self.examples, train_ratio, val_ratio, test_ratio,
                strata_field_name, shuffle)

        splits = tuple(
            Dataset(example_list, self.fields)
            for example_list in (train_data, val_data, test_data)
            if example_list
        )

        # In case the parent sort key isn't none
        if self.sort_key:
            for subset in splits:
                subset.sort_key = self.sort_key

        return splits

    def _get_strata_field_name(self, strata_field_name):
        if strata_field_name is not None:
            return strata_field_name

        for field in self.fields:
            if field.is_target:
                return field.name

        return None


class TabularDataset(Dataset):
    """
    A dataset type for data stored in a single CSV, TSV or JSON file, where
    each row of the file is a single example.
    """

    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        """Creates a TabularDataset from a file containing the data rows and an
        object containing all the fields that we are interested in.

        Parameters
        ----------
            path : str
                Path to the data file.
            format : str
                The format of the data file. Has to be either "CSV", "TSV", or
                "JSON" (case-insensitive).
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
                be in the  same order as the columns in the file). Also, the
                format must be CSV or TSV.

                If type is dict, then it should be a map from the column name
                to the corresponding field/s. Column names not present in
                the dict's keys are ignored. If the format is CSV/TSV,
                then the data file must have a header
                (column names need to be known).
            skip_header : bool
                Whether to skip the first line of the input file.
                If format is CSV/TSV and 'fields' is a dict, then skip_header
                must be False and the data file must have a header.
                Default is False.
            csv_reader_params : dict
                Parameters to pass to the csv reader. Only relevant when
                format is csv or tsv.
                See https://docs.python.org/3/library/csv.html#csv.reader
                for more details.

        Raises
        ------
        ValueError
            If the format given is not one of "CSV", "TSV" or "JSON".
            If fields given as a dict and skip_header is True.
            If format is "JSON" and skip_header is True.
        """

        format = format.lower()

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format in {'csv', 'tsv'}:
                delimiter = ',' if format == "csv" else '\t'
                reader = csv.reader(f, delimiter=delimiter,
                                    **csv_reader_params)
            elif format == "json":
                reader = f
            else:
                raise ValueError(f'Invalid format: {format}')

            # create a list of examples
            examples = create_examples(reader, format, fields, skip_header)

        # we no longer need the column -> field mappings with nested tuples
        # and None values, we just need a flat list of fields
        unpacked_fields = unpack_fields(fields)

        # create a Dataset with lists of examples and fields
        super(TabularDataset, self).__init__(examples, unpacked_fields,
                                             **kwargs)


def create_examples(reader, format, fields, skip_header):
    """Creates a list of examples from the given line reader and fields
    (see TabularDataset.__init__ docs for more info on the fields).

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
    skip_header : bool
        Whether to skip the first line of the input file. (see
        TabularDataset.__init__ docs for more info).

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

    # fromlist is used for CSV/TSV because csv_reader yields data rows as
    # lists, not strings
    make_example_function = {
        "json": Example.fromJSON,
        "csv": Example.fromlist,
        "tsv": Example.fromlist
    }

    if skip_header:
        if format == "json":
            raise ValueError(
                f'When using a {format} file, skip_header must be False.')
        elif format in {"csv", "tsv"} and isinstance(fields, dict):
            raise ValueError(
                f'When using a dict to specify fields with a {format} file,'
                'skip_header must be False and the file must have a header.')

        # skipping the header
        next(reader)

    # if format is CSV/TSV and fields is a dict, transform it to a list
    if format in {"csv", "tsv"} and isinstance(fields, dict):
        # we need a header to know the column names
        header = next(reader)

        # columns not present in the fields dict are ignored (None)
        fields = [fields.get(column, None) for column in header]

    # fields argument is the same for all examples
    make_example = partial(make_example_function[format], fields=fields)

    # map each line from the reader to an example
    examples = map(make_example, reader)

    return list(examples)


def unpack_fields(fields):
    """Flattens the given fields object into a flat list of fields.

    Parameters
    ----------
    fields : (list | dict)
        List or dict that can contain nested tuples and None as values and
        column names as keys (dict).

    Returns
    -------
    list[Field]
        A flat list of Fields found in the given 'fields' object.
    """

    unpacked_fields = list()

    fields = fields.values() if isinstance(fields, dict) else fields

    # None values represent columns that should be ignored
    for field in filter(lambda f: f is not None, fields):
        if isinstance(field, tuple):
            unpacked_fields.extend(field)
        else:
            unpacked_fields.append(field)

    return unpacked_fields


def check_split_ratio(split_ratio):
    """Checks that the split ratio argument is not malformed and if not
    transforms it to a tuple of (train_size, valid_size, test_size) and
    normalizes it if necessary so that all elements sum to 1.

    (See Dataset.split docs for more info).

    Parameters
    ----------
    split_ratio : (float | list[float] | tuple[float])
        The split_ratio should either be a float in the interval (0.0, 1.0)
        (size of train) or a list / tuple of floats of length 2 (or 3) that
        are all larger than 0 and that represent the relative sizes of train,
        (val), test splits.
        If given as a list / tuple, the relative sizes don't have to sum  up
        to 1.0 (they are normalized automatically).

    Returns
    -------
    tuple[float]
        A tuple of (train_size, valid_size, test_size) whose elements sum
        to 1.0.

    Raises
    ------
    ValueError
        If the ratio doesn't obey any of the expected formats described above.
    """

    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        if not (0. < split_ratio < 1.):
            raise ValueError(f'Split ratio {split_ratio} not between 0 and 1')

        train_ratio = split_ratio
        val_ratio = None
        test_ratio = 1.0 - split_ratio
    elif isinstance(split_ratio, list) or isinstance(split_ratio, tuple):
        # A list/tuple of relative ratios is provided
        split_ratio = tuple(split_ratio)
        length = len(split_ratio)

        if length not in {2, 3}:
            raise ValueError(
                f'Split ratio list/tuple should be of length 2 or 3, '
                f'got {length}.')

        for i, ratio in enumerate(split_ratio):
            if float(ratio) <= 0.0:
                raise ValueError(
                    f'Elements of ratio tuple/list must be > 0.0 '
                    f'(got value {ratio} at index {i}).')

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.0:
            split_ratio = tuple(
                float(ratio) / ratio_sum for ratio in split_ratio)

        train_ratio = split_ratio[0]
        if length == 2:
            val_ratio = None
            test_ratio = split_ratio[1]
        else:
            val_ratio = split_ratio[1]
            test_ratio = split_ratio[2]
    else:
        raise ValueError(
            f'Split ratio must be a float, a list or a tuple, '
            f'got {type(split_ratio)}')

    return train_ratio, val_ratio, test_ratio


def rationed_split(examples, train_ratio, val_ratio, test_ratio, shuffle):
    """Splits a list of examples according to the given ratios and returns
    the splits as a tuple of lists (train_examples, valid_examples,
    test_examples).

    The list can also be randomly shuffled before splitting.

    Parameters
    ----------
    examples : list
        A list of examples that is to be split according to the ratios.
    train_ratio : float
        The fraction of examples that should be put into the train split.
    val_ratio : float
        The fraction of examples that should be put into the valid split.
    test_ratio : float
        The fraction of examples that should be put into the test split.
    shuffle : bool
        Whether to shuffle the list before splitting.

    Returns
    -------
    tuple
        The train, valid and test splits, each as a list of examples.

    Raises
    ------
    ValueError
        If the given split ratio is wrong in the sense that it would result
        with at least one empty split.
    """

    # Create a random permutation of examples, then split them
    # by ratio x length slices for each of the train/test/dev? splits
    N = len(examples)

    indices = list(range(N))
    if shuffle:
        random.shuffle(indices)

    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if val_ratio is None:
        if train_len == 0 or (N - train_len) == 0:
            raise ValueError(
                'Bad ratio: both splits should have at least 1 element.')

        indices_tuple = (
            indices[:train_len],
            [],
            indices[train_len:]
        )
    else:
        test_len = int(round(test_ratio * N))
        val_len = N - train_len - test_len

        if train_len * test_len * val_len == 0:
            raise ValueError(
                'Bad ratio: all splits should have at least 1 element.')

        indices_tuple = (
            indices[:train_len],  # Train
            indices[train_len:train_len + val_len],  # Validation
            indices[train_len + val_len:]  # Test
        )

    # Create a tuple of 3 lists, the middle of which is empty if only the
    # train and test ratios were provided
    data = tuple(
        [examples[idx] for idx in indices] for indices in indices_tuple
    )

    return data


def stratified_split(examples, train_ratio, val_ratio, test_ratio,
                     strata_field_name, shuffle):
    """Performs a stratified split on a list of examples according to the
    given ratios and the given strata field.

    Returns the splits as a tuple of lists (train_examples, valid_examples,
    test_examples).

    The list can also be randomly shuffled before splitting.

    Parameters
    ----------
    examples : list
        A list of examples that is to be split according to the ratios.
    train_ratio : float
        The fraction of examples that should be put into the train split.
    val_ratio : float
        The fraction of examples that should be put into the valid split.
    test_ratio : float
        The fraction of examples that should be put into the test split.
    strata_field_name : str
        Name of the field that the examples should be stratified over.
        The values of the strata field have to be hashable.
        Default is 'label' for the conventional label field.
    shuffle : bool
        Whether to shuffle the list before splitting.

    Returns
    -------
    tuple
        The stratified train, valid and test splits, each as a list of
        examples.
    """

    # group the examples by the strata_field
    strata = itertools.groupby(examples,
                               key=lambda ex: getattr(ex, strata_field_name))
    strata = (list(group) for _, group in strata)

    train_split, val_split, test_split = [], [], []
    for group in strata:
        # Split each group of examples according to the ratios given
        group_train_split, group_val_split, group_test_split = rationed_split(
            group,
            train_ratio,
            val_ratio,
            test_ratio,
            shuffle
        )

        # add the group splits to total splits
        train_split += group_train_split
        val_split += group_val_split
        test_split += group_test_split

    # now, for each concrete label value (stratum) - as well as for the whole
    # list of examples - the ratios are preserved
    return train_split, val_split, test_split

class HierarchicalDataset:

    hierarchical_example_tuple = namedtuple("Node",
                                            ['example', 'parent', 'children'])

    # parser(raw example, fields, depth) - returns (parsed example, iterable(raw children))

    def __init__(self, parser, fields):
        self._root_examples = list()
        self._fields = fields
        self._parser = parser

    @staticmethod
    def from_json(dataset, fields, parser):
        ds = HierarchicalDataset(parser, fields)

        root_list = json.loads(dataset)
        ds._load(root_list)

        return ds

    def _load(self, root_examples):
        for root in root_examples:
            self._root_examples.append(self._parse(root, None, 0))

    def _parse(self, raw_object, parent, depth):
        example, raw_children = self._parser(raw_object, self._fields, depth)
        children = [self._parse(c, example, depth + 1) for c in raw_children]
        return HierarchicalDataset.hierarchical_example_tuple(example, parent, children)

    def flatten(self):
        def flatten_node(node, accumulator):
            accumulator.append(node.example)
            for subnode in node.children:
                flatten_node(subnode, accumulator)

        accumulator = list()
        for root_node in self._root_examples:
            flatten_node(root_node, accumulator)

        return accumulator