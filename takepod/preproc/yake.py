import logging

_LOGGER = logging.getLogger(__name__)

try:
    import yake
except ImportError:
    _LOGGER.debug("Problem occured while trying to import yake. "
                  "If the library is not installed visit "
                  "https://github.com/LIAAD/yake for more details.")


class YAKE():
    """Yet Another Keyword Extractor is and unsupervised, corpus-independent, and
       domain and language independent keyword extraction algorithm for extraction from
       single documents. This class is a wrapper of the official implementation available
       at https://github.com/LIAAD/yake."""

    def __init__(self, lang="en", ngram_size=3,
                 dedup_lim=0.9, dedup_func='seqm',
                 windows_size=1, top=20):
        """Constructor that initializes YAKE.

        Parameters
        ----------
        lang : str
            the language of the input text
        ngram_size: int
            maximum ngram size
        dedup_lim: float
            deduplication limit, see yake documentation for details
        dedup_func: str
            deduplication function, see yake documentation for details
        windows_size: int
            context window size
        top: int
            max number of keyphrases to extract
        """

        self._kw_extractor = yake.KeywordExtractor(lan=lang,
                                                   n=ngram_size,
                                                   dedupLim=dedup_lim,
                                                   dedupFunc=dedup_func,
                                                   windowsSize=windows_size,
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
