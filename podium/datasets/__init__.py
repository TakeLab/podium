"""
Package contains datasets.
"""

from .dataset import (
    Dataset,
    DatasetBase,
    DatasetConcatView,
    DatasetIndexedView,
    DatasetSlicedView,
    concat,
    create_view,
    rationed_split,
    stratified_split,
)
from .example_factory import Example, ExampleFactory, ExampleFormat
from .hierarhical_dataset import HierarchicalDataset
from .impl.conllu_dataset import CoNLLUDataset
from .impl.cornell_movie_dialogs import CornellMovieDialogs
from .impl.imdb import IMDB
from .impl.snli import SNLI, SNLISimple
from .impl.sst import SST
from .iterator import BucketIterator, HierarchicalIterator, Iterator, SingleBatchIterator
from .tabular_dataset import TabularDataset


def list_builtin_datasets():
    builtin_datasets = [
        "CornellMovieDialogs",
        "IMDB",
        "SNLI",
        "SNLISimple",
        "SST",
    ]

    print(f"Built-in datasets: {builtin_datasets}")
