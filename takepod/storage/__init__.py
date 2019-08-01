"""Package contains modules for storing and loading datasets and vectors."""

from .example_factory import ExampleFactory
from .field import Field, TokenizedField, MultilabelField, MultioutputField, unpack_fields
from .resources.downloader import (BaseDownloader, SCPDownloader, HttpDownloader,
                                   SimpleHttpDownloader)
from .iterator import (Iterator, SingleBatchIterator, BucketIterator,
                       HierarchicalDatasetIterator)
from .resources.large_resource import LargeResource, SCPLargeResource
from .vectorizers.vectorizer import VectorStorage, BasicVectorStorage
from .vectorizers.tfidf import TfIdfVectorizer
from .vocab import SpecialVocabSymbols, Vocab


# Convention: class imports from same module are continuous in one line until the length
# limit, when they continue in the next line. Imports from subsequent modules are broken
# with newlines.
# The order of imports is the same as the alphabetical order of files in the current
# module, as well as the ordering in `__all__`.
# Function imports follow the same convention, but behind class imports.


__all__ = ["BaseDownloader", "SCPDownloader", "HttpDownloader", "SimpleHttpDownloader",
           "Field", "TokenizedField", "MultilabelField", "MultioutputField",
           "unpack_fields",
           "Iterator", "SingleBatchIterator", "BucketIterator",
           "HierarchicalDatasetIterator", "LargeResource", "SCPLargeResource",
           "VectorStorage", "BasicVectorStorage", "SpecialVocabSymbols", "Vocab",
           "ExampleFactory", "TfIdfVectorizer"]
