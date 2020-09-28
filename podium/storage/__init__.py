"""Package contains modules for storing and loading datasets and vectors."""

from .example_factory import ExampleFactory, ExampleFormat
from .field import (
    Field,
    LabelField,
    MultilabelField,
    MultioutputField,
    TokenizedField,
    unpack_fields,
)
from .resources.downloader import (
    BaseDownloader,
    HttpDownloader,
    SCPDownloader,
    SimpleHttpDownloader,
)
from .resources.large_resource import LargeResource, SCPLargeResource
from .vectorizers.tfidf import TfIdfVectorizer
from .vectorizers.vectorizer import BasicVectorStorage, VectorStorage
from .vocab import SpecialVocabSymbols, Vocab


# Convention: class imports from same module are continuous in one line until the length
# limit, when they continue in the next line. Imports from subsequent modules are broken
# with newlines.
# The order of imports is the same as the alphabetical order of files in the current
# module, as well as the ordering in `__all__`.
# Function imports follow the same convention, but behind class imports.


__all__ = [
    "BaseDownloader",
    "SCPDownloader",
    "HttpDownloader",
    "SimpleHttpDownloader",
    "Field",
    "TokenizedField",
    "LabelField",
    "MultilabelField",
    "MultioutputField",
    "unpack_fields",
    "LargeResource",
    "SCPLargeResource",
    "VectorStorage",
    "BasicVectorStorage",
    "SpecialVocabSymbols",
    "Vocab",
    "ExampleFactory",
    "ExampleFormat",
    "TfIdfVectorizer",
]
