"""Module contains text sentencizer."""
import logging


_LOGGER = logging.getLogger(__name__)


class SpacySentencizer:
    """Detects sentence boundaries and splits the input data on them.
    If the language model is not installed, an attempt is made to install it.
    """

    def __init__(self, language="en"):
        """Sentencizer constructor.

        Parameters
        ----------
        language : str
            Language argument for the sentencizer.
            For the list of supported languages,
            see https://spacy.io/usage/models#languages.
            Default: "en".
        """
        try:
            import spacy

            disable = ["tagger", "ner"]
            nlp = spacy.load(language, disable=disable)
        except OSError:
            _LOGGER.warning(
                f"SpaCy model {language} not found." "Trying to download and install."
            )

            from spacy.cli.download import download

            download(language)
            nlp = spacy.load(language, disable=disable)

        self._nlp = nlp

    def __call__(self, data):
        """Splits the input data on sentence boundaries.

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
