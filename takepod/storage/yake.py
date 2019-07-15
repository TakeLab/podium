import logging

import yake

_LOGGER = logging.getLogger(__name__)


class YAKE():
    """Yet Another Keyword Extractor is and unsupervised, corpus-independent, and
       domain and language independent keyword extraction algorithm for extraction from
       single documents. This class is a wrapper of the official implementation available
       at https://github.com/LIAAD/yake."""

    def __init__(self, lang="en", ngram_size=3, dedup_lim=0.9, dedup_func='seqm',
                 windows_size=1, top=20):
        """Constructor that initializes YAKE.

        Parameters
        ----------
        lang : str
            the language of the input text. If None, defaults to English.
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
                                                   features=None)

    def transform(self, example):
        """Extracts keywords from the example.

        Parameters
        ----------
        example : Example
            dataset example containing a text field
        Returns
        -------
        keywords : list of (str, float)
            list of (keyword, score) sorted ascending by score
        """
        if example is None:
            error_msg = "examples mustn't be None"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)
        raw_text = example.text[0]
        keywords = self._kw_extractor.extract_keywords(raw_text)
        return keywords
