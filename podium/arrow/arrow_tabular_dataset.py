import logging
import tempfile
import itertools
import os
import shutil
import pickle
import csv
from collections import defaultdict
from typing import List, Dict, Union, Tuple, Iterable, Iterator, Any, Callable

from podium.storage import ExampleFactory, Field, unpack_fields
from podium.datasets import Dataset
from podium.storage.example_factory import Example

_LOGGER = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError as ie:
    msg = "Error encountered while importing Pyarrow. If Pyarrow is not installed, " \
          "please visit https://pypi.org/project/pyarrow/ for more information."
    _LOGGER.error(msg)
    raise ie


def _group(iterable, n):
    """ groups an iterable into tuples of size n"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class ArrowDataset:
    """Podium dataset implementation which uses PyArrow as its data storage backend.
    Examples are stored in a file which is then memory mapped for fast random access.
    """

    TEMP_CACHE_FILENAME_PREFIX = 'podium_arrow_cache_'
    CACHE_TABLE_FILENAME = 'podium_arrow_cache.arrow'
    CACHE_FIELDS_FILENAME = 'podium_fields.pkl'

    CHUNK_MAX_SIZE = 10_000

    def __init__(self,
                 table: pa.Table,
                 fields: Union[Dict[str, Field], List[Field]],
                 cache_path: str,
                 mmapped_file: pa.MemoryMappedFile,
                 data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None):
        """Creates a new ArrowDataset instance. Users should use static constructor
        functions like 'from_dataset' to construct new ArrowDataset instances.

        Parameters
        ----------
        table: pyarrow.Table
            Table object that contains example data.

        fields: Union[Dict[str, Field], List[Field]]
            Dict or List of Fields used to create the examples in 'table'.

        cache_path: str
            Path to the directory where the cache file is saved.

        mmapped_file: pyarrow.MemoryMappedFile
            Open MemoryMappedFile descriptor of the cache file.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.
        """
        self.cache_path = cache_path
        self.fields = unpack_fields(fields)
        self.field_dict = {field.name: field for field in fields}
        self.mmapped_file = mmapped_file
        self.table = table
        self.data_types = data_types

    @staticmethod
    def from_dataset(dataset: Dataset,
                     cache_path: str = None,
                     data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None
                     ) -> 'ArrowDataset':
        """Creates an ArrowDataset instance from a podium.datasets.Dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used to create an ArrowDataset.

        cache_path: str
            Path to the directory where the cache file will saved.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.

        Returns
        -------
        ArrowDataset
            ArrowDataset instance created from the passed Dataset.

        """
        return ArrowDataset.from_examples(dataset.fields,
                                          iter(dataset),
                                          cache_path,
                                          data_types)

    @staticmethod
    def from_examples(fields: Union[Dict[str, Field], List[Field]],
                      examples: Iterable[Example],
                      cache_path: str = None,
                      data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None
                      ) -> 'ArrowDataset':
        """ Creates an ArrowDataset from the provided Examples.

        Parameters
        ----------
        fields: Union[Dict[str, Field], List[Field]]
            Dict or List of Fields used to create the Examples.

        examples: Iterable[Example]
            Iterable of examples.

        cache_path: str
            Path to the directory where the cache file will saved.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.

        Returns
        -------
        ArrowDataset
            ArrowDataset instance created from the passed Examples.
        """

        fields = unpack_fields(fields)

        if cache_path is None:
            cache_path = tempfile.mkdtemp(prefix=ArrowDataset.TEMP_CACHE_FILENAME_PREFIX)

        # dump dataset table
        cache_table_path = os.path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)

        # TODO hande cache case when cache is present

        chunks_iter = _group(examples, ArrowDataset.CHUNK_MAX_SIZE)

        # get first group to infer schema
        first_group = next(chunks_iter)
        record_batch = ArrowDataset._examples_to_recordbatch(first_group,
                                                             fields,
                                                             data_types)
        inferred_data_types = ArrowDataset._schema_to_data_types(record_batch.schema)

        # check for missing data types in inferred schema
        ArrowDataset._check_for_missing_data_types(fields, inferred_data_types)

        # write cache file to disk
        with pa.OSFile(cache_table_path, 'wb') as f:
            with pa.RecordBatchFileWriter(f, schema=record_batch.schema) as writer:
                writer.write(record_batch)  # write first chunk
                for examples_chunk in chunks_iter:  # write rest of chunks
                    record_batch = \
                        ArrowDataset._examples_to_recordbatch(examples_chunk,
                                                              fields,
                                                              inferred_data_types)
                    writer.write(record_batch)

        mmapped_file = pa.memory_map(cache_table_path)
        table = pa.RecordBatchFileReader(mmapped_file).read_all()

        return ArrowDataset(table, fields, cache_path, mmapped_file, inferred_data_types)

    @staticmethod
    def from_tabular_file(path: str,
                          format: str,
                          fields: Union[Dict[str, Field], List[Field]],
                          cache_path: str = None,
                          data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None,
                          skip_header: bool = False,
                          csv_reader_params: Dict = None) -> 'ArrowDataset':
        """Loads a tabular file format (csv, tsv, json) as an ArrowDataset.

        Parameters
        ----------
        path : str
                Path to the data file.

        format : str
                The format of the data file. Has to be either "CSV", "TSV", or
                "JSON" (case-insensitive).

        fields : Union[Dict[str, Field], List[Field]]
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

        cache_path: str
            Path to the directory where the cache file will saved.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.

        skip_header : bool
                Whether to skip the first line of the input file.
                If format is CSV/TSV and 'fields' is a dict, then skip_header
                must be False and the data file must have a header.
                Default is False.

        csv_reader_params : Dict
                Parameters to pass to the csv reader. Only relevant when
                format is csv or tsv.
                See https://docs.python.org/3/library/csv.html#csv.reader
                for more details.

        Returns
        -------
        ArrowDataset
            ArrowDataset instance containing the examples from the tabular file.

        """
        format = format.lower()
        csv_reader_params = {} if csv_reader_params is None else csv_reader_params

        with open(os.path.expanduser(path), encoding="utf8") as f:
            if format in {'csv', 'tsv'}:
                delimiter = ',' if format == "csv" else '\t'
                reader = csv.reader(f, delimiter=delimiter,
                                    **csv_reader_params)
            elif format == "json":
                reader = f
            else:
                error_msg = "Invalid format: {}".format(format)
                _LOGGER.error(error_msg)
                raise ValueError(error_msg)

            if skip_header:
                if format == "json":
                    error_msg = "When using a {} file, skip_header must be False." \
                        .format(format)
                    _LOGGER.error(error_msg)
                    raise ValueError(error_msg)
                elif format in {"csv", "tsv"} and isinstance(fields, dict):
                    error_msg = "When using a dict to specify fields with a {}" \
                                " file, skip_header must be False and the file must " \
                                "have a header.".format(format)
                    _LOGGER.error(error_msg)
                    raise ValueError(error_msg)

                # skipping the header
                next(reader)

            # if format is CSV/TSV and fields is a dict, transform it to a list
            if format in {"csv", "tsv"} and isinstance(fields, dict):
                # we need a header to know the column names
                header = next(reader)

                # columns not present in the fields dict are ignored (None)
                fields = [fields.get(column, None) for column in header]

            # fields argument is the same for all examples
            # fromlist is used for CSV/TSV because csv_reader yields data rows as
            # lists, not strings
            example_factory = ExampleFactory(fields)
            make_example_function = {
                "json": example_factory.from_json,
                "csv": example_factory.from_list,
                "tsv": example_factory.from_list
            }

            make_example = make_example_function[format]

            # map each line from the reader to an example
            examples = map(make_example, reader)
            return ArrowDataset.from_examples(fields,
                                              examples,
                                              cache_path=cache_path,
                                              data_types=data_types)

    @staticmethod
    def _schema_to_data_types(inferred_schema: pa.Schema
                              ) -> Dict[str, Tuple[pa.DataType, pa.DataType]]:
        """Converts a pyarrow Schema instance into the ArrowDataset data_types format.

        Parameters
        ----------
        inferred_schema: pyarrow.Schema
            Schema to be converted into the ArrowDataset data_type format.

        Returns
        -------
        Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            ArrowDataset data_types extracted from the passed Schema instance.
        """
        dtypes = defaultdict(dict)

        for dtype_field in inferred_schema:
            field_name, part = dtype_field.name.rsplit('_', 1)
            dtypes[field_name][part] = dtype_field.type

        return {field_name: (field_dtypes.get('raw'), field_dtypes.get('tokenized'))
                for field_name, field_dtypes in dtypes.items()}

    @staticmethod
    def _examples_to_recordbatch(examples: List[Example],
                                 fields: Union[Dict[str, Field], List[Field]],
                                 data_types=None) -> pa.RecordBatch:
        """Converts an list of examples into a pyarrow RecordBatch object.

        Parameters
        ----------
        examples: List[Example]
            Examples to transform into a RecordBatch.

        fields: Union[Dict[str, Field], List[Field]]
            Dict or List of Fields used to create the Examples.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.

        Returns
        -------
        pyarrow.RecordBatch
            RecordBatch object containing the data from the passed Examples.
        """
        data_type_override = {}
        if data_types is not None:
            for field_name, (raw_dtype, tokenized_dtype) in data_types.items():
                if raw_dtype is not None:
                    raw_field_name = field_name + "_raw"
                    data_type_override[raw_field_name] = raw_dtype

                if tokenized_dtype is not None:
                    tokenized_field_name = field_name + "_tokenized"
                    data_type_override[tokenized_field_name] = tokenized_dtype

        arrays = []
        array_names = []
        for field in fields:
            field_raw_column = []
            field_tokenized_column = []

            for example in examples:
                raw, tokenized = getattr(example, field.name)
                field_raw_column.append(raw)
                field_tokenized_column.append(tokenized)

            raw_fieldname = field.name + "_raw"
            dtype = data_type_override.get(raw_fieldname)  # None if not overridden
            array = pa.array(field_raw_column, type=dtype)
            if array.type != pa.null():
                arrays.append(array)
                array_names.append(raw_fieldname)

            tokenized_fieldname = field.name + "_tokenized"
            dtype = data_type_override.get(tokenized_fieldname)  # None if not overridden
            array = pa.array(field_tokenized_column, type=dtype)
            if array.type != pa.null():
                arrays.append(array)
                array_names.append(tokenized_fieldname)

        return pa.RecordBatch.from_arrays(arrays, names=array_names)

    @staticmethod
    def _recordbatch_to_examples(record_batch, fields) -> Iterator[Example]:
        """Converts a pyarrow RecordBatch object into Podium examples.

        Parameters
        ----------
        record_batch: pyarrow.RecordBatch
            RecordBatch object containing the Example data.

        fields: Union[Dict[str, Field], List[Field]]
            Dict or List of Fields used to create the Examples.

        Returns
        -------
        Iterator[podium.storage.Example]
            An Iterator iterating over the Examples contained in the passed RecordBatch.

        """
        fields = unpack_fields(fields)
        fieldnames = tuple(field.name for field in fields)
        field_value_iterators = tuple(ArrowDataset._field_values(record_batch, fieldname)
                                      for fieldname in fieldnames)

        for row in zip(*field_value_iterators):
            example = Example(fieldnames)
            for fieldname, values in zip(fieldnames, row):
                setattr(example, fieldname, values)
            yield example

    @staticmethod
    def _check_for_missing_data_types(fields, data_types):
        """ Checks if data_types has non-null data types for every field in fields.
        Raises a RuntimeError if not.

        Parameters
        ----------
        fields: List[Field]
            Fields to check for.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Inferred data_types to check for missing data types.

        Raises
        ------
        RuntimeError
            If not every field in fields has appropriate data types in data_types.

        """
        for field in fields:
            raw_dtype, tokenized_dtype = data_types.get(field.name, (None, None))

            error_part = None
            if raw_dtype is None and field.store_as_raw:
                error_part = "raw"

            if tokenized_dtype is None and (field.store_as_tokenized or field.tokenize):
                error_part = "tokenized"

            if error_part is not None:
                msg = "Data type of the {} part of field '{}' cannot be inferred. " \
                      "Please provide the explicit " \
                      "pyarrow datatype trough the 'data_type' argument. The data_type " \
                      "format is " \
                      "{{field_name: (raw_dtype, tokenized_dtype)}}." \
                    .format(error_part, field.name)
                _LOGGER.error(msg)
                raise RuntimeError(msg)

    @staticmethod
    def load_cache(cache_path) -> 'ArrowDataset':
        """ Loads a cached ArrowDataset contained in the cache_path directory.
        Fields will be loaded into memory but the Example data will be memory mapped
        avoiding unnecessary memory usage.

        Parameters
        ----------
        cache_path: str
            Path to the directory where the ArrowDataset cache is contained.

        Returns
        -------
        ArrowDataset
            the ArrowDataset loaded from the passed cache directory.

        """
        # load fields
        fields_file_path = os.path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        with open(fields_file_path, 'rb') as fields_cache_file:
            fields = pickle.load(fields_cache_file)

        # load dataset as memory mapped arrow table
        table_file_path = os.path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)
        mmapped_file = pa.memory_map(table_file_path)
        table = pa.RecordBatchFileReader(mmapped_file).read_all()
        return ArrowDataset(table, fields, cache_path, mmapped_file)

    def dump_cache(self,
                   cache_path: str = None) -> str:
        """ Saves this dataset at cache_path. Dumped datasets can be loaded with the
        ArrowDataset.load_cache static method. All fields contained in this dataset must
        be serializable using pickle.

        Parameters
        ----------
        cache_path: str
            Path to the directory where the dataset is to be dumped.
            Can be None, in which case a temporary directory will be created and used to
            dump the cache. The chosen cache dir is always returned.

        Returns
        -------
        str
            The chosen cache directory path. Useful when cache_path is None and a
            temporary directory is created.

        """
        if cache_path == self.cache_path:
            msg = "Cache path same as datasets cache path. " \
                  "Dataset can't overwrite its own cache. Cache path: {}" \
                .format(cache_path)
            _LOGGER.error(msg)
            raise Exception(msg)

        if cache_path is None:
            cache_path = tempfile.mkdtemp(prefix=ArrowDataset.TEMP_CACHE_FILENAME_PREFIX)

        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)

        # pickle fields
        cache_fields_path = os.path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        with open(cache_fields_path, 'wb') as fields_cache_file:
            pickle.dump(self.fields, fields_cache_file)

        # dump table
        cache_table_path = os.path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)
        with pa.OSFile(cache_table_path, 'wb') as f:
            with pa.RecordBatchFileWriter(f, self.table.schema) as writer:
                writer.write(self.table)

        return cache_path

    def as_dataset(self) -> Dataset:
        """ Loads this ArrowDataset into memory and returns an Dataset object containing
        the loaded data.

        Returns
        -------
        Dataset
            Dataset containing all examples of this ArrowDataset.
        """
        examples = list(ArrowDataset._recordbatch_to_examples(self.table, self.fields))
        return Dataset(examples, self.fields)

    def batch(self):
        """Creates an input and target batch containing the whole dataset.
       The format of the batch is the same as the batches returned by the Iterator class.

       Returns
       -------
       input_batch, target_batch
               Two objects containing the input and target batches over
               the whole dataset.
        """
        # TODO custom batch method?
        return self.as_dataset().batch()

    @staticmethod
    def _field_values(record_batch: pa.RecordBatch,
                      fieldname: str) -> Iterable[Tuple[Any, Any]]:
        """ Iterates over the raw and tokenized values of a field contained in the
        record_batch.

        Parameters
        ----------
        record_batch: pyarrow.RecordBatch
            RecordBatch containing example data.

        fieldname: str
            Name of the field whose raw and tokenized data is to be iterated over.

        Returns
        -------
        Iterable[Tuple[raw, tokenized]]
            Iterable over the (raw, tokenized) values of the provided fieldname.
        """

        record_batch_fieldnames = tuple(field.name for field in record_batch.schema)

        columnname_raw = fieldname + "_raw"
        if columnname_raw in record_batch_fieldnames:
            raw_values = (value.as_py() for value in record_batch[columnname_raw])
        else:
            raw_values = itertools.repeat(None, len(record_batch))

        columnname_tokenized = fieldname + "_tokenized"
        if columnname_tokenized in record_batch_fieldnames:
            tokenized_values = (
                value.as_py() for value in record_batch[columnname_tokenized]
            )
        else:
            tokenized_values = itertools.repeat(None, len(record_batch))

        return zip(raw_values, tokenized_values)

    def __getitem__(self, item,
                    deep_copy=False) -> Union[Example, 'ArrowDataset']:
        """Returns an example or a new ArrowDataset containing the indexed examples.
        If indexed with an int, only the example at that position will be returned.
        If Indexed with a slice or iterable, all examples indexed by the object
        will be collected and a new dataset containing only those examples will be
        returned.

        Examples in the returned Dataset are the same ones present in the
        original dataset. If a complete deep-copy of the dataset, or its slice,
        is needed please refer to the `get` method.

        Usage example:

            example = dataset[1] # Indexing by single integer returns a single example

            new_dataset = dataset[1:10:2] # Multiindexing returns a new dataset containing
                                          # the indexed examples.

            new_dataset_2 = dataset[(1,5,6,9)] # Returns a new dataset containing the
                                               # indexed Examples.

        Parameters
        ----------
        item: int or slice or iterable
            Index used to index examples.

        deep_copy: bool
            Not used.

        Returns
        -------
        Example or Dataset
            If i is an int, a single example will be returned.
            If i is a slice, list or tuple, a copy of this dataset containing
            only the indexed examples will be returned.
        """

        if isinstance(item, int):
            # slices extract row, indexing with int extracts column
            record_batch = self.table[item:item + 1]
            example_iter = ArrowDataset \
                ._recordbatch_to_examples(record_batch, self.fields)
            return next(example_iter)  # returns the one example

        if isinstance(item, slice):
            table_slice = self.table[item]

        else:
            if isinstance(item, tuple):
                item = list(item)
            table_slice = self.table.take(item)

        return ArrowDataset(table=table_slice,
                            fields=self.fields,
                            cache_path=self.cache_path,
                            mmapped_file=self.mmapped_file)

    def __len__(self) -> int:
        """ Returns the number of Examples in this Dataset.

        Returns
        -------
        int
            The number of Examples in this Dataset.
        """
        return len(self.table)

    def __iter__(self) -> Iterator[Example]:
        """ Iterates over Examples in this dataset.

        Returns
        -------
        Iterator[Example]
            Iterator over all the examples in this dataset.
        """
        yield from self._recordbatch_to_examples(self.table, self.fields)

    def __getattr__(self, fieldname) -> Iterator[Tuple[Any, Any]]:
        """ Iterates over the raw and tokenized values of all examples in this dataset.

        Parameters
        ----------
        fieldname: str
            Name of the field whose values are to be iterated over.

        Returns
        -------
        Iterator[Tuple[raw, tokenized]]
            Iterator over the raw and tokenized values of all examples in this dataset.

        """
        if fieldname in self.field_dict:
            return ArrowDataset._field_values(self.table, fieldname)

        else:
            error_msg = "Dataset has no field {}.".format(fieldname)
            _LOGGER.error(error_msg)
            raise AttributeError(error_msg)

    def filter(self,
               predicate: Callable[[Example], bool]) -> 'ArrowDataset':
        """ Creates a new ArrowDataset instance containing only Examples for which the
        predicate returns True.

        Parameters
        ----------
        predicate : Callable[[Example], bool]
            Callable used as a filtering predicate. It takes an Example as a parameter
            and returns True if the Example is to be accepted, and False otherwise.

        Returns
        -------
        ArrowDataset
            New ArrowDataset containing Filtered Examples.
        """
        indices = [i for i, example in enumerate(self) if predicate(example)]
        return self[indices]

    def sorted(self, key: Callable[[Example], Any], reverse=False) -> 'ArrowDataset':
        """Returns a new ArrowDataset with sorted Examples.

        Parameters
        ----------
        key: Callable[[Example], Any]
            Extracts a comparable value from an Example.
            That value will be used to determine Example ordering.

        reverse: bool
            If True, the returned dataset will be reversed.

        Returns
        -------
        ArrowDataset
            An ArrowDataset containing sorted Examples from this dataset.
        """
        def index_key(i, _dataset=self):
            return key(_dataset[i])

        indices = list(range(len(self)))
        indices.sort(key=index_key, reverse=reverse)
        return self[indices]

    def close(self):
        """ Closes resources held by the ArrowDataset."""
        if self.mmapped_file is not None:
            self.mmapped_file.close()
            self.mmapped_file = None

        else:  # Do nothing
            msg = "Attempted closing an already closed ArrowDataset."
            _LOGGER.debug(msg)

    def delete_cache(self):
        """ Deletes the cache directory."""
        if self.mmapped_file is not None:
            self.close()
        shutil.rmtree(self.cache_path)

    def finalize_fields(self, *datasets):
        """ Builds vocabularies of all the non-eager fields in the dataset,
        from the Dataset objects given as \\*args and then finalizes all the
        fields.

        Parameters
        ----------
        \\*args
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
                filter(lambda arg: isinstance(arg, Dataset), datasets))

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
