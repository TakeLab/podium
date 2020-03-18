"""Package contains concrete datasets"""

from .pauza_dataset import PauzaHRDataset
from .catacx_dataset import CatacxDataset
from .imdb_sentiment_dataset import IMDB
from .eurovoc_dataset import EuroVocDataset
from .cornell_movie_dialogs_dataset import CornellMovieDialogsConversationalDataset
from .snli_dataset import SNLIDataset, SNLISimple


__all__ = ["IMDB", "CatacxDataset",
           "CornellMovieDialogsConversationalDataset", "EuroVocDataset",
           "PauzaHRDataset", "SNLIDataset", "SNLISimple"]
