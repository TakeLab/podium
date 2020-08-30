import pyarrow as pa
import tempfile
import itertools
from os import path
from collections import defaultdict

from podium.datasets import Dataset
from podium.storage.example_factory import Example


def group(n, iterable):
    # groups an iterable into tuples of size n
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class ArrowDataset(Dataset):
    CACHE_FILENAME = 'arrow_cache'
    CHUNK_SIZE = 100

    def __init__(self, fields, examples, cache_path=None, data_types=None):

        self.fields = fields
        self.fields_dict = {field.name: field for field in fields}

        if cache_path is None:
            self.temp_dir = tempfile.TemporaryDirectory(prefix='podium_arrow_cache')
            self.cache_file_path = path.join(self.temp_dir.name, ArrowDataset.CACHE_FILENAME)

        else:
            self.temp_dir = None
            self.cache_file_path = path.join(cache_path, ArrowDataset.CACHE_FILENAME)

        # TODO hande cache case when cache is present

        chunks_iter = group(ArrowDataset.CHUNK_SIZE, examples)

        # get first group to infer schema
        first_group = chunks_iter.__next__()
        record_batch = self._examples_to_recordbatch(first_group)

        with pa.OSFile(self.cache_file_path, 'wb') as f:
            with pa.RecordBatchFileWriter(f, schema=record_batch.schema) as writer:
                writer.write(record_batch)  # write first chunk
                for examples_chunk in chunks_iter:  # write rest of chunks
                    record_batch = self._examples_to_recordbatch(examples_chunk)
                    writer.write(record_batch)

        self.mmapped_file = pa.memory_map(self.cache_file_path)
        self.table: pa.Table = pa.RecordBatchFileReader(self.mmapped_file).read_all()

    def _examples_to_recordbatch(self, examples):
        #  transpose examples
        #  TODO add explicit typing
        arrays = []
        array_names = []
        for field in self.fields:
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

    def _recordbatch_to_examples(self, record_batch):
        fieldnames = tuple(field.name for field in self.fields)
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

    def __getitem__(self, item, deep_copy=False):

        if isinstance(item, int):
            record_batch = self.table[item:item + 1]  # indexing with int extracts column
            return self._recordbatch_to_examples(record_batch)[0]

        if isinstance(item, slice):
            record_batch = self.table[item]

        else:
            record_batch = self.table.take(item)

        examples = self._recordbatch_to_examples(record_batch)
        return Dataset(examples, self.fields)

    def __len__(self):
        return self.table.num_rows

    def __iter__(self):
        for i in len(self):
            yield self[i]

    # TODO __getattr__

    def close(self):
        self.mmapped_file.close()
        if self.temp_dir is not None:
            self.temp_dir.close()
            self.temp_dir = None

        # TODO close file handles
