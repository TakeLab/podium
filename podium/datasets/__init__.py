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
from .impl.cornell_movie_dialogs_dataset import CornellMovieDialogsConversationalDataset
from .impl.imdb_sentiment_dataset import IMDB
from .impl.snli_dataset import SNLIDataset, SNLISimple
from .impl.sst_sentiment_dataset import SST
from .iterator import BucketIterator, HierarchicalIterator, Iterator, SingleBatchIterator
from .tabular_dataset import TabularDataset


def list_builtin_datasets():
    builtin_datasets = [
        "CornellMovieDialogsConversationalDataset",
        "IMDB",
        "SNLIDataset",
        "SNLISimple",
        "SST",
    ]

    print(f"Built-in datasets: {builtin_datasets}")
