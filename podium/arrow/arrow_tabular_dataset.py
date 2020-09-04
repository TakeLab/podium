import pyarrow as pa
import tempfile
import itertools
from os import path
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


class ArrowDataset(Dataset):
    TEMP_CACHE_FILENAME_PREFIX = 'podium_arrow_cache'
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
    def _load_from_cache_path(cache_path):
        # load fields
        fields_file_path = path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        with open(fields_file_path, 'rb') as fields_cache_file:
            fields = pickle.load(fields_cache_file)

        # load dataset as memory mapped arrow table
        table_file_path = path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)
        mmapped_file = pa.memory_map(table_file_path)
        table = pa.RecordBatchFileReader(mmapped_file).read_all()
        return ArrowDataset(table, fields, cache_path, mmapped_file)

    @staticmethod
    def from_examples(fields, examples, cache_path=None, data_types=None):

        if cache_path is None:
            cache_path = tempfile.mkdtemp(prefix=ArrowDataset.TEMP_CACHE_FILENAME_PREFIX)

        # pickle fields
        cache_fields_path = path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        with open(cache_fields_path, 'wb') as fields_cache_file:
            pickle.dump(fields, fields_cache_file)

        # dump dataset table
        cache_table_path = path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)

        # TODO hande cache case when cache is present

        chunks_iter = group(examples, ArrowDataset.CHUNK_MAX_SIZE)

        # get first group to infer schema
        first_group = chunks_iter.__next__()
        record_batch = ArrowDataset._examples_to_recordbatch(first_group, fields)

        with pa.OSFile(cache_table_path, 'wb') as f:
            with pa.RecordBatchFileWriter(f, schema=record_batch.schema) as writer:
                writer.write(record_batch)  # write first chunk
                for examples_chunk in chunks_iter:  # write rest of chunks
                    record_batch = ArrowDataset._examples_to_recordbatch(examples_chunk, fields)
                    writer.write(record_batch)

        return ArrowDataset(cache_path, fields)

    @staticmethod
    def _examples_to_recordbatch(examples, fields):
        #  transpose examples
        #  TODO add explicit typing
        arrays = []
        array_names = []
        for field in fields:
            field_raw_column = []
            field_tokenized_column = []

            for example in examples:
                raw, tokenized = getattr(example, field.name)
                field_raw_column.append(raw)
                field_tokenized_column.append(tokenized)

            arrays.append(pa.array(field_raw_column))
            array_names.append(field.name + "_raw")

            arrays.append(pa.array(field_tokenized_column))
            array_names.append(field.name + "_tokenized")

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

    def dump_cache(self, cache_path):

        # pickle fields
        cache_fields_path = path.join(cache_path, ArrowDataset.CACHE_FIELDS_FILENAME)
        with open(cache_fields_path, 'wb') as fields_cache_file:
            pickle.dump(self.fields, fields_cache_file)

        # dump table
        cache_table_path = path.join(cache_path, ArrowDataset.CACHE_TABLE_FILENAME)
        with pa.OSFile(cache_table_path, 'wb') as f:
            with pa.RecordBatchFileWriter(f, schema=self.table.schema) as writer:
                writer.write(self.table)

    def __getitem__(self, item, deep_copy=False):

        if isinstance(item, int):
            record_batch = self.table[item:item + 1]  # slices extract row, indexing with int extracts column
            return ArrowDataset._recordbatch_to_examples(record_batch, self.fields)[0]

        if isinstance(item, slice):
            table_slice = self.table[item]

        else:
            table_slice = self.table.take(item)

        return ArrowDataset(table=table_slice,
                            fields=self.fields,
                            cache_path=self.cache_path,
                            mmapped_file=self.mmapped_file)

    def __len__(self):
        return self.table.num_rows

    def __iter__(self):
        for i in len(self):
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
