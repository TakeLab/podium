import pyarrow as pa
import tempfile
import itertools
import os
import shutil
import logging
import pickle
import io
import csv

from podium.storage import ExampleFactory
from podium.datasets import Dataset
from podium.storage.example_factory import Example

_LOGGER = logging.getLogger(__name__)


def group(iterable, n):
    """ groups an iterable into tuples of size n"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class ArrowDataset:
    TEMP_CACHE_FILENAME_PREFIX = 'podium_arrow_cache_'
    CACHE_TABLE_FILENAME = 'podium_arrow_cache.arrow'
    CACHE_FIELDS_FILENAME = 'podium_fields.pkl'

    CHUNK_MAX_SIZE = 10000

    def __init__(self, table, fields, cache_path, mmapped_file, data_types=None):
        self.cache_path = cache_path

        self.fields = fields
        self.field_dict = {field.name: field for field in fields}

        self.mmapped_file = mmapped_file

        self.table = table

        self.data_types = data_types

    @staticmethod
    def from_dataset(dataset, cache_path=None, data_types=None):
        return ArrowDataset.from_examples(dataset.fields, iter(dataset), cache_path, data_types)

    @staticmethod
    def from_examples(fields, examples, cache_path=None, data_types: dict = None):

        if cache_path is None:
            cache_path = tempfile.mkdtemp(prefix=ArrowDataset.TEMP_CACHE_FILENAME_PREFIX)

        # dump dataset table
        cache_table_path = os.path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)

        # TODO hande cache case when cache is present

        chunks_iter = group(examples, ArrowDataset.CHUNK_MAX_SIZE)

        # get first group to infer schema
        first_group = next(chunks_iter)
        record_batch = ArrowDataset._examples_to_recordbatch(first_group, fields, data_types)

        with pa.OSFile(cache_table_path, 'wb') as f:
            with pa.RecordBatchFileWriter(f, schema=record_batch.schema) as writer:
                writer.write(record_batch)  # write first chunk
                for examples_chunk in chunks_iter:  # write rest of chunks
                    record_batch = ArrowDataset._examples_to_recordbatch(examples_chunk, fields, data_types)
                    writer.write(record_batch)

        mmapped_file = pa.memory_map(cache_table_path)
        table = pa.RecordBatchFileReader(mmapped_file).read_all()

        return ArrowDataset(table, fields, cache_path, mmapped_file, data_types)

    @staticmethod
    def from_tabular_file(path, format, fields, cache_path=None, data_types=None, skip_header=False,
                 csv_reader_params=None):

        format = format.lower()
        csv_reader_params = {} if csv_reader_params is None else csv_reader_params

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
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
                error_msg = "When using a {} file, skip_header must be False.".format(format)
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
            return ArrowDataset.from_examples(fields, examples, cache_path=cache_path, data_types=data_types)

    @staticmethod
    def _examples_to_recordbatch(examples, fields, data_types=None):
        data_type_override = {} if data_types is None else data_types
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
    def _recordbatch_to_examples(record_batch, fields):
        fieldnames = tuple(field.name for field in fields)
        field_value_iterators = tuple(ArrowDataset._field_values(record_batch, fieldname) for fieldname in fieldnames)

        for row in zip(*field_value_iterators):
            example = Example(fieldnames)
            for fieldname, values in zip(fieldnames, row):
                setattr(example, fieldname, values)
            yield example

    @staticmethod
    def load_cache(cache_path):
        # load fields
        fields_file_path = os.path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        with open(fields_file_path, 'rb') as fields_cache_file:
            fields = pickle.load(fields_cache_file)

        # load dataset as memory mapped arrow table
        table_file_path = os.path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)
        mmapped_file = pa.memory_map(table_file_path)
        table = pa.RecordBatchFileReader(mmapped_file).read_all()
        return ArrowDataset(table, fields, cache_path, mmapped_file)

    def dump_cache(self, cache_path=None):

        if cache_path == self.cache_path:
            msg = "Cache path same as datasets cache path. " \
                  "Dataset can't overwrite its own cache. Cache path: {}".format(cache_path)
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

    def as_dataset(self):
        examples = list(ArrowDataset._recordbatch_to_examples(self.table, self.fields))
        return Dataset(examples, self.fields)

    def batch(self):
        # TODO custom batch method?
        return self.as_dataset().batch()

    @staticmethod
    def _field_values(record_batch, fieldname):

        def infinite_generator(value):
            while True:
                yield value

        record_batch_fieldnames = tuple(field.name for field in record_batch.schema)

        columnname_raw = fieldname + "_raw"
        if columnname_raw in record_batch_fieldnames:
            raw_values = (value.as_py() for value in record_batch[columnname_raw])
        else:
            raw_values = infinite_generator(None)

        columnname_tokenized = fieldname + "_tokenized"
        if columnname_tokenized in record_batch_fieldnames:
            tokenized_values = (value.as_py() for value in record_batch[columnname_tokenized])
        else:
            tokenized_values = infinite_generator(None)

        return zip(raw_values, tokenized_values)

    def __getitem__(self, item, deep_copy=False):

        if isinstance(item, int):
            record_batch = self.table[item:item + 1]  # slices extract row, indexing with int extracts column
            example_iter = ArrowDataset._recordbatch_to_examples(record_batch, self.fields)
            return next(example_iter)  # returns the one example

        if isinstance(item, slice):
            table_slice = self.table[item]

        else:
            table_slice = self.table.take(item)

        return ArrowDataset(table=table_slice,
                            fields=self.fields,
                            cache_path=self.cache_path,
                            mmapped_file=self.mmapped_file)

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        yield from self._recordbatch_to_examples(self.table, self.fields)

    def __getattr__(self, fieldname):
        if fieldname in self.field_dict:
            return ArrowDataset._field_values(self.table, fieldname)

        else:
            error_msg = "Dataset has no field {}.".format(fieldname)
            _LOGGER.error(error_msg)
            raise AttributeError(error_msg)

    def filter(self, predicate):
        indices = [i for i, example in enumerate(self) if predicate(example)]
        return self[indices]

    def close(self):
        if self.mmapped_file is not None:
            self.mmapped_file.close()
            self.mmapped_file = None

        else:  # Do nothing
            msg = "Attempted closing an already closed ArrowDataset."
            _LOGGER.debug(msg)

    def delete_cache(self):
        if self.mmapped_file is not None:
            self.close()
        shutil.rmtree(self.cache_path)

    def finalize_fields(self, *args):
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
