"""Package contains datasets"""

from .dataset import Dataset, stratified_split, rationed_split
from .hierarhical_dataset import HierarchicalDataset
from .tabular_dataset import TabularDataset
from .impl.pauza_dataset import PauzaHRDataset
from .impl.catacx_dataset import CatacxDataset
from .impl.imdb_sentiment_dataset import IMDB
from .impl.eurovoc_dataset import EuroVocDataset
from .impl.sst_sentiment_dataset import SST
from .impl.cornell_movie_dialogs_dataset import CornellMovieDialogsConversationalDataset
from .impl.snli_dataset import SNLIDataset, SNLISimple
from .iterator import (Iterator, SingleBatchIterator, BucketIterator,
                       HierarchicalDatasetIterator)

__all__ = ["Dataset", "TabularDataset", "HierarchicalDataset",
           "stratified_split", "rationed_split",
           "IMDB", "CatacxDataset", "SST",
           "CornellMovieDialogsConversationalDataset", "EuroVocDataset", "PauzaHRDataset",
           "Iterator", "SingleBatchIterator", "BucketIterator",
           "HierarchicalDatasetIterator", "SNLIDataset", "SNLISimple"]