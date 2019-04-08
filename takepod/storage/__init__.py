"""Package contains modules for storing and loading datasets and vectors."""

from .dataset import Dataset, HierarchicalDataset, \
    TabularDataset, stratified_split, rationed_split
from .downloader import (BaseDownloader, SCPDownloader, HttpDownloader,
                         SimpleHttpDownloader)
from .example import Example
from .field import Field, TokenizedField, MultilabelField
from .iterator import Iterator, BucketIterator, HierarchicalDatasetIterator
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
           "Example",
           "Field", "TokenizedField", "MultilabelField",
           "Iterator", "BucketIterator", "HierarchicalDatasetIterator",
           "LargeResource", "SCPLargeResource",
           "VectorStorage", "BasicVectorStorage",
           "SpecialVocabSymbols", "Vocab",
           "stratified_split", "rationed_split",
           "ExampleFactory"]
