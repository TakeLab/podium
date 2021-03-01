"""
Home of the TakeLab Podium project. Podium is a Python machine learning library
that helps users to accelerate use of natural language processing models.

See http://takelab.fer.hr/podium/ for complete documentation.
"""
import importlib
import os
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any


__version__ = "1.1.0"


def _is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None


_import_structure = {
    "field": ["Field", "LabelField", "MultilabelField", "MultioutputField"],
    "vocab": ["Vocab", "Special", "BOS", "EOS", "UNK", "PAD"],
    "datasets.dataset": [
        "Dataset",
        "DatasetBase",
        "DatasetConcatView",
        "DatasetIndexedView",
        "DatasetSlicedView",
        "concat",
        "create_view",
        "rationed_split",
        "stratified_split",
    ],
    "datasets.example_factory": ["Example", "ExampleFactory", "ExampleFormat"],
    "datasets.hierarhical_dataset": ["HierarchicalDataset"],
    "datasets.impl.catacx_dataset": ["CatacxDataset"],
    "datasets.impl.conllu_dataset": ["CoNLLUDataset"],
    "datasets.impl.cornell_movie_dialogs_dataset": [
        "CornellMovieDialogsConversationalDataset"
    ],
    "datasets.impl.eurovoc_dataset": ["EuroVocDataset"],
    "datasets.impl.imdb_sentiment_dataset": ["IMDB"],
    "datasets.impl.pauza_dataset": ["PauzaHRDataset"],
    "datasets.impl.snli_dataset": ["SNLIDataset", "SNLISimple"],
    "datasets.impl.sst_sentiment_dataset": ["SST"],
    "datasets.iterator": [
        "BucketIterator",
        "HierarchicalDatasetIterator",
        "Iterator",
        "SingleBatchIterator",
    ],
    "datasets.tabular_dataset": ["TabularDataset"],
    "preproc.functional": ["remove_stopwords", "truecase"],
    "preproc.hooks": [
        "MosesNormalizer",
        "NLTKStemmer",
        "RegexReplace",
        "SpacyLemmatizer",
        "TextCleanUp",
    ],
    "preproc.lemmatizer": ["CroatianLemmatizer"],
    "preproc.sentencizers": ["SpacySentencizer"],
    "preproc.stemmer": ["CroatianStemmer"],
    "preproc.tokenizers": ["get_tokenizer"],
    "storage.resources.downloader": [
        "BaseDownloader",
        "HttpDownloader",
        "SCPDownloader",
        "SimpleHttpDownloader",
    ],
    "storage.resources.large_resource": ["LargeResource", "SCPLargeResource"],
    "vectorizers.impl": ["GloVe", "NlplVectorizer"],
    "vectorizers.tfidf": ["TfIdfVectorizer"],
    "vectorizers.vectorizer": [
        "BasicVectorStorage",
        "VectorStorage",
        "random_normal_default_vector",
        "zeros_default_vector",
    ],
}

if _is_package_available("pyarrow"):
    _import_structure["datasets.arrow"] = ["DiskBackedDataset"]
if _is_package_available("datasets"):
    _import_structure["datasets.hf"] = [
        "convert_features_to_fields",
        "HFDatasetConverter",
    ]
if _is_package_available("yake"):
    _import_structure["preproc.yake"] = ["YAKE"]


if TYPE_CHECKING:
    from .datasets.dataset import (
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
    from .datasets.example_factory import Example, ExampleFactory, ExampleFormat
    from .datasets.hierarhical_dataset import HierarchicalDataset
    from .datasets.impl.catacx_dataset import CatacxDataset
    from .datasets.impl.conllu_dataset import CoNLLUDataset
    from .datasets.impl.cornell_movie_dialogs_dataset import (
        CornellMovieDialogsConversationalDataset,
    )
    from .datasets.impl.eurovoc_dataset import EuroVocDataset
    from .datasets.impl.imdb_sentiment_dataset import IMDB
    from .datasets.impl.pauza_dataset import PauzaHRDataset
    from .datasets.impl.snli_dataset import SNLIDataset, SNLISimple
    from .datasets.impl.sst_sentiment_dataset import SST
    from .datasets.iterator import (
        BucketIterator,
        HierarchicalDatasetIterator,
        Iterator,
        SingleBatchIterator,
    )
    from .datasets.tabular_dataset import TabularDataset
    from .field import Field, LabelField, MultilabelField, MultioutputField
    from .preproc.functional import remove_stopwords, truecase
    from .preproc.hooks import (
        MosesNormalizer,
        NLTKStemmer,
        RegexReplace,
        SpacyLemmatizer,
        TextCleanUp,
    )
    from .preproc.lemmatizer import CroatianLemmatizer
    from .preproc.sentencizers import SpacySentencizer
    from .preproc.stemmer import CroatianStemmer
    from .preproc.tokenizers import get_tokenizer
    from .storage.resources.downloader import (
        BaseDownloader,
        HttpDownloader,
        SCPDownloader,
        SimpleHttpDownloader,
    )
    from .storage.resources.large_resource import LargeResource, SCPLargeResource
    from .vectorizers.impl import GloVe, NlplVectorizer
    from .vectorizers.tfidf import TfIdfVectorizer
    from .vectorizers.vectorizer import (
        BasicVectorStorage,
        VectorStorage,
        random_normal_default_vector,
        zeros_default_vector,
    )
    from .vocab import Vocab

    if _is_package_available("pyarrow"):
        from .datasets.arrow import DiskBackedDataset
    if _is_package_available("datasets"):
        from .datasets.hf import HFDatasetConverter, convert_features_to_fields
    if _is_package_available("yake"):
        from .preproc.yake import YAKE
else:

    class _LazyModule(ModuleType):
        """
        Module class that surfaces all objects but only performs associated
        imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        # This code is inspired by:
        # https://github.com/huggingface/transformers/blob/master/src/huggingface/__init__.py
        def __init__(self, name, import_structure):
            super().__init__(name)
            self._modules = set(import_structure.keys())
            self._class_to_module = {}
            for key, values in import_structure.items():
                for value in values:
                    self._class_to_module[value] = key
            # Needed for autocompletion in an IDE
            self.__all__ = list(import_structure.keys()) + sum(
                import_structure.values(), []
            )

        # Needed for autocompletion in an IDE
        def __dir__(self):
            return super().__dir__() + self.__all__

        def __getattr__(self, name: str) -> Any:
            if name == "__version__":
                return __version__
            elif name in self._modules:
                value = self._get_module(name)
            elif name in self._class_to_module.keys():
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            else:
                raise AttributeError(f"module {self.__name__} has no attribute {name}")

            setattr(self, name, value)
            return value

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
