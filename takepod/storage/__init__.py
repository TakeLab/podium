"""Package contains modules for storing and loading datasets and vectors."""

from .dataset import Dataset, HierarchicalDataset, \
    TabularDataset, stratified_split, rationed_split, unpack_fields
from .downloader import (BaseDownloader, SCPDownloader, HttpDownloader,
                         SimpleHttpDownloader)
from .field import Field, TokenizedField, MultilabelField, MultioutputField
from .iterator import Iterator, SingleBatchIterator,BucketIterator, HierarchicalDatasetIterator
from .large_resource import LargeResource, SCPLargeResource
from .vectorizer import VectorStorage, BasicVectorStorage
from .vocab import SpecialVocabSymbols, Vocab
from .example_factory import ExampleFactory

# Convention: class imports from same module are continuous in one line until the length
# limit, when they continue in the next line. Imports from subsequent modules are broken
# with newlines.
# The order of imports is the same as the alphabetical order of files in the current
# module, as well as the ordering in `__all__`.
# Function imports follow the same convention, but behind class imports.


__all__ = ["Dataset", "HierarchicalDataset", "TabularDataset",
           "BaseDownloader", "SCPDownloader", "HttpDownloader", "SimpleHttpDownloader",
           "Field", "TokenizedField", "MultilabelField", "MultioutputField",
           "unpack_fields",
           "Iterator", "SingleBatchIterator", "BucketIterator", "HierarchicalDatasetIterator",
           "LargeResource", "SCPLargeResource",
           "VectorStorage", "BasicVectorStorage",
           "SpecialVocabSymbols", "Vocab",
           "stratified_split", "rationed_split",
           "ExampleFactory"]
