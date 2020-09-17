"""This module contains wrapper for Yet Another Keyword Extractor library."""
import logging

_LOGGER = logging.getLogger(__name__)

try:
    import yake
except ImportError as e:
    _LOGGER.error("Problem occured while trying to import yake. "
                  "If the library is not installed visit "
                  "https://github.com/LIAAD/yake for more details.")
    raise e


class YAKE:
    """Yet Another Keyword Extractor is an unsupervised, corpus-independent, and
       domain and language independent keyword extraction algorithm for extraction from
       single documents. This class is a wrapper of the official implementation available
       at https://github.com/LIAAD/yake."""

    def __init__(self,
                 lan: str = "en",
                 n: int = 3,
                 dedupLim: float = 0.8,
                 dedupFunc: str = 'levenshtein',
                 windowsSize: int = 2,
                 top: int = 20):
        """Constructor that initializes YAKE.

        Parameters
        ----------
        lan : str
            the language of the input text
        n: int
            maximum ngram size
        dedupLim: float
            deduplication limit, see yake documentation for details
        dedupFunc: str
            deduplication function, see yake documentation for details
        windowsSize: int
            context window size
        top: int
            max number of keyphrases to extract
        """

        self._kw_extractor = yake.KeywordExtractor(lan=lan,
                                                   n=n,
                                                   dedupLim=dedupLim,
                                                   dedupFunc=dedupFunc,
                                                   windowsSize=windowsSize,
                                                   top=top,
                                                   features=None)  # features dict

    def __call__(self, string):
        """Extracts keywords from the string. See transform function for details.
        """
        return self.transform(string)

    def transform(self, string):
        """Extracts keywords from the string.

        Parameters
        ----------
        string : str
            source text for keyword extraction
        Returns
        -------
        keywords : list of str
            list of keywords sorted ascending by score (lower score is better)
        """
        keywords = self._kw_extractor.extract_keywords(string)
        return [kw for kw, _ in keywords]
