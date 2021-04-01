import csv
import itertools
import os
import pickle
import shutil
import tempfile
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from podium.field import Field, unpack_fields

from .dataset import DatasetBase, _pandas_to_examples
from .example_factory import Example, ExampleFactory


try:
    import pyarrow as pa
except ImportError:
    print(
        "Error encountered while importing Pyarrow. If Pyarrow is not installed, "
        "please visit https://pypi.org/project/pyarrow/ for more information."
    )
    raise


def _chunkify(iterable, n):
    """
    Splits an iterable into chunks of size n.
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


TEMP_CACHE_FILENAME_PREFIX = "podium_arrow_cache_"
CACHE_TABLE_FILENAME = "podium_arrow_cache.arrow"
CACHE_FIELDS_FILENAME = "podium_fields.pkl"


class DiskBackedDataset(DatasetBase):
    """
    Podium dataset implementation which uses PyArrow as its data storage
    backend.

    Examples are stored in a file which is then memory mapped for fast random
    access.
    """

    def __init__(
        self,
        table: pa.Table,
        fields: Union[Dict[str, Field], List[Field]],
        cache_path: Optional[str],
        mmapped_file: pa.MemoryMappedFile,
        data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None,
    ):
        """
        Creates a new DiskBackedDataset instance. Users should use static
        constructor functions like 'from_dataset' to construct new
        DiskBackedDataset instances.

        Parameters
        ----------
        table: pyarrow.Table
            Table object that contains example data.

        fields: Union[Dict[str, Field], List[Field]]
            Dict or List of Fields used to create the examples in 'table'.

        cache_path: Optional[str]
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
        self.mmapped_file = mmapped_file
        self.table = table
        self.data_types = data_types
        super().__init__(fields)

    @staticmethod
    def from_dataset(
        dataset: DatasetBase,
        cache_path: Optional[str] = None,
        data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None,
    ) -> "DiskBackedDataset":
        """
        Creates a DiskBackedDataset instance from a podium.datasets.DatasetBase
        instance.

        Parameters
        ----------
        dataset: DatasetBase
            DatasetBase instance to be used to create the DiskBackedDataset.

        cache_path: Optional[str]
            Path to the directory where the cache file will saved.
            The whole directory will be used as the cache and will be deleted
            when `delete_cache` is called. It is recommended to create a new
            directory to use exclusively as the cache, or to leave this as None.

            If None, a temporary directory will be created.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.

        Returns
        -------
        DiskBackedDataset
            DiskBackedDataset instance created from the passed DatasetBase instance.
        """
        return DiskBackedDataset.from_examples(
            dataset.fields, iter(dataset), cache_path, data_types
        )

    @staticmethod
    def from_examples(
        fields: Union[Dict[str, Field], List[Field], Tuple[Field]],
        examples: Iterable[Example],
        cache_path: Optional[str] = None,
        data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None,
        chunk_size=1024,
    ) -> "DiskBackedDataset":
        """
        Creates a DiskBackedDataset from the provided Examples.

        Parameters
        ----------
        fields: Union[Dict[str, Field], List[Field]]
            Dict or List of Fields used to create the Examples.

        examples: Iterable[Example]
            Iterable of examples.

        cache_path: Optional[str]
            Path to the directory where the cache file will saved.
            The whole directory will be used as the cache and will be deleted
            when `delete_cache` is called. It is recommended to create a new
            directory to use exclusively as the cache, or to leave this as None.

            If None, a temporary directory will be created.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.

        chunk_size: int
            Maximum number of examples to be loaded before dumping to the on-disk cache
            file. Use lower number if memory usage is an issue while loading.

        Returns
        -------
        DiskBackedDataset
            DiskBackedDataset instance created from the passed Examples.
        """

        fields = unpack_fields(fields)

        if cache_path is None:
            cache_path = tempfile.mkdtemp(prefix=TEMP_CACHE_FILENAME_PREFIX)

        # dump dataset table
        cache_table_path = os.path.join(cache_path, CACHE_TABLE_FILENAME)

        # TODO hande cache case when cache is present

        chunks_iter = _chunkify(examples, chunk_size)

        # get first chunk to infer schema
        first_chunk = next(chunks_iter)
        record_batch = DiskBackedDataset._examples_to_recordbatch(
            first_chunk, fields, data_types
        )
        inferred_data_types = DiskBackedDataset._schema_to_data_types(record_batch.schema)

        # check for missing data types in inferred schema
        DiskBackedDataset._check_for_missing_data_types(fields, inferred_data_types)

        # write cache file to disk
        with pa.OSFile(cache_table_path, "wb") as f:
            with pa.RecordBatchFileWriter(f, schema=record_batch.schema) as writer:
                writer.write(record_batch)  # write first chunk
                for examples_chunk in chunks_iter:  # write rest of chunks
                    record_batch = DiskBackedDataset._examples_to_recordbatch(
                        examples_chunk, fields, inferred_data_types
                    )
                    writer.write(record_batch)

        mmapped_file = pa.memory_map(cache_table_path)
        table = pa.RecordBatchFileReader(mmapped_file).read_all()

        return DiskBackedDataset(
            table, fields, cache_path, mmapped_file, inferred_data_types
        )

    @staticmethod
    def from_tabular_file(
        path: str,
        format: str,
        fields: Union[Dict[str, Field], List[Field]],
        cache_path: Optional[str] = None,
        data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None,
        chunk_size=10_000,
        skip_header: bool = False,
        delimiter=None,
        csv_reader_params: Dict = None,
    ) -> "DiskBackedDataset":
        """
        Loads a tabular file format (csv, tsv, json) as a DiskBackedDataset.

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

        cache_path: Optional[str]
            Path to the directory where the cache file will saved.
            The whole directory will be used as the cache and will be deleted
            when `delete_cache` is called. It is recommended to create a new
            directory to use exclusively as the cache, or to leave this as None.

            If None, a temporary directory will be created.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.

        chunk_size: int
            Maximum number of examples to be loaded before dumping to the on-disk cache
            file. Use lower number if memory usage is an issue while loading.

        skip_header : bool
                Whether to skip the first line of the input file.
                If format is CSV/TSV and 'fields' is a dict, then skip_header
                must be False and the data file must have a header.
                Default is False.
        delimiter: str
            Delimiter used to separate columns in a row.
            If set to None, the default delimiter for the given format will
            be used.
        csv_reader_params : Dict
                Parameters to pass to the csv reader. Only relevant when
                format is csv or tsv.
                See https://docs.python.org/3/library/csv.html#csv.reader
                for more details.

        Returns
        -------
        DiskBackedDataset
            DiskBackedDataset instance containing the examples from the tabular file.
        """
        format = format.lower()
        csv_reader_params = {} if csv_reader_params is None else csv_reader_params

        with open(os.path.expanduser(path), encoding="utf8") as f:
            if format in {"csv", "tsv"}:
                delimiter = "," if format == "csv" else "\t"
                reader = csv.reader(f, delimiter=delimiter, **csv_reader_params)
            elif format == "json":
                reader = f
            else:
                raise ValueError(f"Invalid format: {format}")

            if skip_header:
                if format == "json":
                    raise ValueError(
                        f"When using a {format} file, skip_header must be False."
                    )
                elif format in {"csv", "tsv"} and isinstance(fields, dict):
                    raise ValueError(
                        f"When using a dict to specify fields with a {format} "
                        "file, skip_header must be False and the file must "
                        "have a header."
                    )

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
                "tsv": example_factory.from_list,
            }

            make_example = make_example_function[format]

            # map each line from the reader to an example
            example_iterator = map(make_example, reader)
            return DiskBackedDataset.from_examples(
                fields,
                example_iterator,
                cache_path=cache_path,
                data_types=data_types,
                chunk_size=chunk_size,
            )

    @staticmethod
    def _schema_to_data_types(
        inferred_schema: pa.Schema,
    ) -> Dict[str, Tuple[pa.DataType, pa.DataType]]:
        """
        Converts a pyarrow Schema instance into the DiskBackedDataset data_types
        format.

        Parameters
        ----------
        inferred_schema: pyarrow.Schema
            Schema to be converted into the DiskBackedDataset data_type format.

        Returns
        -------
        Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            DiskBackedDataset data_types extracted from the passed Schema instance.
        """
        dtypes = defaultdict(dict)

        for dtype_field in inferred_schema:
            field_name, part = dtype_field.name.rsplit("_", 1)
            dtypes[field_name][part] = dtype_field.type

        return {
            field_name: (field_dtypes.get("raw"), field_dtypes.get("tokenized"))
            for field_name, field_dtypes in dtypes.items()
        }

    @staticmethod
    def _examples_to_recordbatch(
        examples: List[Example],
        fields: Union[Dict[str, Field], List[Field]],
        data_types=None,
    ) -> pa.RecordBatch:
        """
        Converts an list of examples into a pyarrow RecordBatch object.

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
                raw, tokenized = example[field.name]
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
        """
        Converts a pyarrow RecordBatch object into Podium examples.

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
        field_value_iterators = tuple(
            DiskBackedDataset._field_values(record_batch, fieldname)
            for fieldname in fieldnames
        )

        for row in zip(*field_value_iterators):
            example = Example()
            for fieldname, values in zip(fieldnames, row):
                example[fieldname] = values
            yield example

    @staticmethod
    def _check_for_missing_data_types(fields, data_types):
        """
        Checks if data_types has non-null data types for every field in fields.
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
            if raw_dtype is None and field._keep_raw:
                error_part = "raw"

            if tokenized_dtype is None:
                error_part = "tokenized"

            if error_part is not None:
                raise RuntimeError(
                    f"Data type of the {error_part} part of field '{field.name}' cannot be inferred. "
                    "Please provide the explicit "
                    "pyarrow datatype trough the 'data_type' argument. The data_type "
                    "format is "
                    "{{field_name: (raw_dtype, tokenized_dtype)}}."
                )

    @staticmethod
    def load_cache(cache_path) -> "DiskBackedDataset":
        """
        Loads a cached DiskBackedDataset contained in the cache_path directory.
        Fields will be loaded into memory but the Example data will be memory
        mapped avoiding unnecessary memory usage.

        Parameters
        ----------
        cache_path: Optional[str]
            Path to the directory where the cache file will be saved.
            The whole directory will be used as the cache and will be deleted
            when `delete_cache` is called. It is recommended to create a new
            directory to use exclusively as the cache, or to leave this as None.

            If None, a temporary directory will be created.

        Returns
        -------
        DiskBackedDataset
            the DiskBackedDataset loaded from the passed cache directory.
        """
        # load fields
        fields_file_path = os.path.join(cache_path, CACHE_FIELDS_FILENAME)
        with open(fields_file_path, "rb") as fields_cache_file:
            fields = pickle.load(fields_cache_file)

        # load dataset as memory mapped arrow table
        table_file_path = os.path.join(cache_path, CACHE_TABLE_FILENAME)
        mmapped_file = pa.memory_map(table_file_path)
        table = pa.RecordBatchFileReader(mmapped_file).read_all()
        return DiskBackedDataset(table, fields, cache_path, mmapped_file)

    def dump_cache(self, cache_path: Optional[str] = None) -> str:
        """
        Saves this dataset at cache_path. Dumped datasets can be loaded with the
        DiskBackedDataset.load_cache static method. All fields contained in this
        dataset must be serializable using pickle.

        Parameters
        ----------
        cache_path: Optional[str]
            Path to the directory where the cache file will saved.
            The whole directory will be used as the cache and will be deleted
            when `delete_cache` is called. It is recommended to create a new
            directory to use exclusively as the cache, or to leave this as None.

            If None, a temporary directory will be created.

        Returns
        -------
        str
            The chosen cache directory path. Useful when cache_path is None and a
            temporary directory is created.
        """
        if cache_path == self.cache_path:
            raise ValueError(
                "Cache path same as datasets cache path. "
                f"Dataset can't overwrite its own cache. Cache path: {cache_path}"
            )

        if cache_path is None:
            cache_path = tempfile.mkdtemp(prefix=TEMP_CACHE_FILENAME_PREFIX)

        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)

        # pickle fields
        cache_fields_path = os.path.join(cache_path, CACHE_FIELDS_FILENAME)
        with open(cache_fields_path, "wb") as fields_cache_file:
            pickle.dump(self.fields, fields_cache_file)

        # dump table
        cache_table_path = os.path.join(cache_path, CACHE_TABLE_FILENAME)
        with pa.OSFile(cache_table_path, "wb") as f:
            with pa.RecordBatchFileWriter(f, self.table.schema) as writer:
                writer.write(self.table)

        return cache_path

    def _get_examples(self) -> List[Example]:
        """
        Loads this DiskBackedDataset into memory and returns a list containing
        the loaded Examples.
        """
        return list(DiskBackedDataset._recordbatch_to_examples(self.table, self.fields))

    @staticmethod
    def _field_values(
        record_batch: pa.RecordBatch, fieldname: str
    ) -> Iterable[Tuple[Any, Any]]:
        """
        Iterates over the raw and tokenized values of a field contained in the
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

    def __getitem__(
        self, item: Union[int, Iterable[int], slice]
    ) -> Union[Example, "DiskBackedDataset"]:
        """
        Returns an example or a new DiskBackedDataset containing the indexed
        examples. If indexed with an int, only the example at that position will
        be returned. If Indexed with a slice or iterable, all examples indexed
        by the object will be collected and a new dataset containing only those
        examples will be returned.

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

        Returns
        -------
        Example or Dataset
            If i is an int, a single example will be returned.
            If i is a slice, list or tuple, a copy of this dataset containing
            only the indexed examples will be returned.
        """

        if isinstance(item, int):
            # slices extract row, indexing with int extracts column
            record_batch = self.table[item : item + 1]
            example_iter = DiskBackedDataset._recordbatch_to_examples(
                record_batch, self.fields
            )
            return next(example_iter)  # returns the one example

        if isinstance(item, slice):
            table_slice = self.table[item]

        else:
            if isinstance(item, tuple):
                item = list(item)
            table_slice = self.table.take(item)

        return DiskBackedDataset(
            table=table_slice,
            fields=self.fields,
            cache_path=self.cache_path,
            mmapped_file=self.mmapped_file,
        )

    def __len__(self) -> int:
        """
        Returns the number of Examples in this Dataset.

        Returns
        -------
        int
            The number of Examples in this Dataset.
        """
        return len(self.table)

    def __iter__(self) -> Iterator[Example]:
        """
        Iterates over Examples in this dataset.

        Returns
        -------
        Iterator[Example]
            Iterator over all the examples in this dataset.
        """
        yield from self._recordbatch_to_examples(self.table, self.fields)

    def __del__(self):
        if self.mmapped_file is not None:
            self.close()

    def close(self):
        """
        Closes resources held by the DiskBackedDataset.

        Only closes the cache file handle. The cache will not be deleted from
        disk. For cache deletion, use `delete_cache`.
        """
        if self.mmapped_file is not None:
            self.mmapped_file.close()
            self.mmapped_file = None
            self.table = None

        else:
            warnings.warn("Attempted closing an already closed DiskBackedDataset.")

    def delete_cache(self):
        """
        Deletes the cache directory and all cache files belonging to this
        dataset.

        After this call is executed, any DiskBackedDataset created by
        slicing/indexing this dataset and any view over this dataset will not be
        usable any more. Any dataset created from this dataset should be dumped
        to a new directory before calling this method.
        """
        if self.mmapped_file is not None:
            self.close()

        shutil.rmtree(self.cache_path)

    @classmethod
    def from_pandas(
        cls,
        df,
        fields: Union[Dict[str, Field], List[Field]],
        index_field: Optional[Field] = None,
        cache_path: Optional[str] = None,
        data_types: Dict[str, Tuple[pa.DataType, pa.DataType]] = None,
        chunk_size: int = 1024,
    ) -> "DiskBackedDataset":
        """
        Creates a DiskBackedDataset instance from a pandas Dataframe.

        Parameters
        ----------
        df: pandas.Dataframe
            Pandas dataframe from which data will be taken.

        fields: Union[Dict[str, Field], List[Field]]
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

        cache_path: Optional[str]
            Path to the directory where the cache file will saved.
            The whole directory will be used as the cache and will be deleted
            when `delete_cache` is called. It is recommended to create a new
            directory to use exclusively as the cache, or to leave this as None.

            If None, a temporary directory will be created.

        data_types: Dict[str, Tuple[pyarrow.DataType, pyarrow.DataType]]
            Dictionary mapping field names to pyarrow data types. This is required when a
            field can have missing data and the data type can't be inferred. The data type
            tuple has two values, corresponding to the raw and tokenized data types in an
            example. None can be used as a wildcard data type and will be overridden by an
            inferred data type if possible.

        chunk_size: int
            Maximum number of examples to be loaded before dumping to the on-disk cache
            file. Use lower number if memory usage is an issue while loading.

        Returns
        -------
        Dataset
            Dataset containing data from the Dataframe
        """

        example_iterator = _pandas_to_examples(df, fields, index_field=index_field)
        if isinstance(fields, dict):
            fields = [index_field] + list(fields.values())
        else:
            fields = [index_field] + fields

        return DiskBackedDataset.from_examples(
            fields, example_iterator, cache_path, data_types, chunk_size
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["mmapped_file"]
        return state

    def __setstate__(self, state):
        mmapped_file = pa.memory_map(
            os.path.join(state["cache_path"], CACHE_TABLE_FILENAME)
        )
        state["mmapped_file"] = mmapped_file
        self.__dict__ = state
