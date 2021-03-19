"""
Package contains modules for preprocessing.
"""

from .hooks import (
    KeywordExtractor,
    MosesNormalizer,
    NLTKStemmer,
    RegexReplace,
    SpacyLemmatizer,
    TextCleanUp,
    as_posttokenize_hook,
    remove_stopwords,
    truecase,
)
from .lemmatizer import CroatianLemmatizer
from .sentencizers import SpacySentencizer
from .stemmer import CroatianStemmer
from .tokenizers import get_tokenizer
