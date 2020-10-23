"""Package contains concrete datasets"""

from .catacx_dataset import CatacxDataset
from .conllu_dataset import CoNLLUDataset
from .cornell_movie_dialogs_dataset import CornellMovieDialogsConversationalDataset
from .eurovoc_dataset import EuroVocDataset
from .imdb_sentiment_dataset import IMDB
from .iris_dataset import IrisDataset
from .pauza_dataset import PauzaHRDataset
from .snli_dataset import SNLIDataset, SNLISimple
from .sst_sentiment_dataset import SST


__all__ = [
    "IMDB",
    "CatacxDataset",
    "CoNLLUDataset",
    "CornellMovieDialogsConversationalDataset",
    "EuroVocDataset",
    "PauzaHRDataset",
    "SNLIDataset",
    "SNLISimple",
    "SST",
    "IrisDataset",
]
