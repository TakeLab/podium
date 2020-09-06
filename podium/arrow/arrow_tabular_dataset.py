import pyarrow as pa
import tempfile
import itertools
from os import path, mkdir
import shutil
import logging
import pickle

from podium.datasets import Dataset
from podium.storage.example_factory import Example

_LOGGER = logging.getLogger(__name__)


def group(iterable, n):
    # groups an iterable into tuples of size n
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

    CHUNK_MAX_SIZE = 1000

    def __init__(self, table, fields, cache_path, mmapped_file):
        self.cache_path = cache_path

        self.fields = fields
        self.fields_dict = {field.name: field for field in fields}

        self.mmapped_file = mmapped_file

        self.table = table

    @staticmethod
    def from_dataset(dataset, cache_path=None, data_types=None):
        return ArrowDataset.from_examples(dataset.fields, dataset.examples, cache_path, data_types)

    @staticmethod
    def from_examples(fields, examples, cache_path=None, data_types: dict = None):

        data_types_override = {}
        if data_types is not None:
            for field_name, (raw_dtype, tokenized_dtype) in data_types.items():
                if raw_dtype is not None:
                    raw_field_name = field_name + "_raw"
                    data_types_override[raw_field_name] = raw_dtype

                if tokenized_dtype is not None:
                    tokenized_field_name = field_name + "_tokenized"
                    data_types_override[tokenized_field_name] = tokenized_dtype

        if cache_path is None:
            cache_path = tempfile.mkdtemp(prefix=ArrowDataset.TEMP_CACHE_FILENAME_PREFIX)

        # pickle fields
        # cache_fields_path = path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        # with open(cache_fields_path, 'wb') as fields_cache_file:
        #     pickle.dump(fields, fields_cache_file)

        # dump dataset table
        cache_table_path = path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)

        # TODO hande cache case when cache is present

        chunks_iter = group(examples, ArrowDataset.CHUNK_MAX_SIZE)

        # get first group to infer schema
        first_group = chunks_iter.__next__()
        record_batch = ArrowDataset._examples_to_recordbatch(first_group, fields, data_types_override)

        with pa.OSFile(cache_table_path, 'wb') as f:
            with pa.RecordBatchFileWriter(f, schema=record_batch.schema) as writer:
                writer.write(record_batch)  # write first chunk
                for examples_chunk in chunks_iter:  # write rest of chunks
                    record_batch = ArrowDataset._examples_to_recordbatch(examples_chunk, fields, data_types_override)
                    writer.write(record_batch)

        return ArrowDataset.load_cache(cache_path)

    @staticmethod
    def _examples_to_recordbatch(examples, fields, data_types=None):
        #  transpose examples
        #  TODO add explicit typing
        data_types = {} if data_types is None else data_types
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
            dtype = data_types.get(raw_fieldname)  # None if not overridden
            array = pa.array(field_raw_column, type=dtype)
            arrays.append(array)
            array_names.append(raw_fieldname)

            tokenized_fieldname = field.name + "_tokenized"
            dtype = data_types.get(tokenized_fieldname)  # None if not overridden
            array = pa.array(field_tokenized_column, type=dtype)
            arrays.append(array)
            array_names.append(tokenized_fieldname)

        return pa.RecordBatch.from_arrays(arrays, names=array_names)

    @staticmethod
    def _recordbatch_to_examples(record_batch, fields):
        fieldnames = tuple(field.name for field in fields)
        examples = [Example(fieldnames) for _ in range(len(record_batch))]

        for fieldname in fieldnames:
            columnname_raw = fieldname + "_raw"
            raw_column = record_batch[columnname_raw]

            columnname_tokenized = fieldname + "_tokenized"
            tokenized_column = record_batch[columnname_tokenized]

            for example, raw, tokenized in zip(examples, raw_column, tokenized_column):
                data = (raw.as_py(), tokenized.as_py())  # (raw, tokenized)
                setattr(example, fieldname, data)

        return examples

    @staticmethod
    def load_cache(cache_path):
        # load fields
        fields_file_path = path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        with open(fields_file_path, 'rb') as fields_cache_file:
            fields = pickle.load(fields_cache_file)

        # load dataset as memory mapped arrow table
        table_file_path = path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)
        mmapped_file = pa.memory_map(table_file_path)
        table = pa.RecordBatchFileReader(mmapped_file).read_all()
        return ArrowDataset(table, fields, cache_path, mmapped_file)

    def dump_cache(self, cache_path):

        if not path.isdir(cache_path):
            mkdir(cache_path)

        # pickle fields
        cache_fields_path = path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        with open(cache_fields_path, 'wb') as fields_cache_file:
            pickle.dump(self.fields, fields_cache_file)

        # dump table
        cache_table_path = path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)
        with pa.OSFile(cache_table_path, 'wb') as f:
            with pa.RecordBatchFileWriter(f, schema=self.table.schema) as writer:
                writer.write(self.table)

    def as_dataset(self):
        examples = ArrowDataset._recordbatch_to_examples(self.table, self.fields)
        return Dataset(examples, self.fields)

    def batch(self):
        return self.as_dataset().batch()  # TODO custom batch method?

    def __getitem__(self, item, deep_copy=False):

        if isinstance(item, int):
            record_batch = self.table[item:item + 1]  # slices extract row, indexing with int extracts column
            return ArrowDataset._recordbatch_to_examples(record_batch, self.fields)[0]

        if isinstance(item, slice):
            # TODO doesn't work if step !=1 , causes SIGSEGV on dump
            # write
            table_slice = self.table[item]

        else:
            # TODO doesn't work, doesn't select rows, causes SIGSEGV on dump
            table_slice = self.table.take(item)

        return ArrowDataset(table=table_slice,
                            fields=self.fields,
                            cache_path=self.cache_path,
                            mmapped_file=self.mmapped_file)

    def __len__(self):
        return self.table.num_rows

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # TODO __getattr__

    def close(self):
        if self.mmapped_file is not None:
            self.mmapped_file.close()
            self.mmapped_file = None

        else:  # Do nothing
            msg = "Attempted closing an already closed ArrowDataset."
            _LOGGER.debug(msg)

        # TODO close file handles

    def delete_cache(self):
        if self.mmapped_file is not None:
            self.close()
        shutil.rmtree(self.cache_path)
