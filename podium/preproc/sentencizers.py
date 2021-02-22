"""
Module contains text sentencizer.
"""
from podium.utils.general_utils import load_spacy_model_or_raise


class SpacySentencizer:
    """
    Detects sentence boundaries and splits the input data on them.

    If the language model is not installed, an attempt is made to install it.
    """

    def __init__(self, language="en"):
        """
        Sentencizer constructor.

        Parameters
        ----------
        language : str
            Language argument for the sentencizer.
            For the list of supported languages,
            see https://spacy.io/usage/models#languages.
            Default: "en".
        """
        language = "en_core_web_sm" if language == "en" else language
        self._nlp = load_spacy_model_or_raise(language, disable=["ner"])

    def __call__(self, data):
        """
        Splits the input data on sentence boundaries.

        Parameters
        ----------
        data
            The input data.

        Returns
        -------
        list of str
            The input data split on sentences boundaries.
        """
        return [sent.text for sent in self._nlp(data).sents]
