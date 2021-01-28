"""
Package contains modules for preprocessing.
"""

from .hooks import (
    MosesNormalizer,
    NLTKStemmer,
    RegexReplace,
    SpacyLemmatizer,
    TextCleanUp,
    remove_stopwords,
    truecase,
)
from .lemmatizer import CroatianLemmatizer
from .sentencizers import SpacySentencizer
from .stemmer import CroatianStemmer
from .tokenizers import get_tokenizer
