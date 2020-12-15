"""Module contains functional preprocessing hooks."""
import warnings
from typing import Callable, List, Tuple


def truecase(oov: str = "title") -> Callable[[str], str]:
    """Returns a pretokenization hook that applies truecasing to the raw textual data.

    To use this hook, the truecase library has to be installed.

    Parameters
    ----------
    oov : str
        Defines how to handle out of vocabulary tokens not seen while training
        the truecasing model. 3 options are supported:
        title - returns OOV tokens in 'title' format
        lower - returns OOV tokens in lower case
        as-is - returns OOV tokens as is

        Default is 'title'.

    Returns
    -------
    callable
        Function that truecases the raw data.

    Raises
    ------
    ImportError
        If the truecase library is not installed.
    """

    try:
        import truecase as truecase_
    except ImportError:
        print(
            "Problem occured while trying to import truecase. "
            "If the library is not installed visit "
            "https://pypi.org/project/truecase/ for more details."
        )
        raise

    if oov not in {"title", "lower", "as-is"}:
        raise ValueError("Specified out of vocabulary option is not supported")

    def _truecase_hook(raw):
        return truecase_.get_true_case(raw, out_of_vocabulary_token_option=oov)

    return _truecase_hook


def remove_stopwords(
    language: str = "en",
) -> Callable[[str, List[str]], Tuple[str, List[str]]]:
    """Returns a posttokenization hook that removes stop words
    from the tokenized textual data. The raw part is left unchanged.

    Stop words are obtained from the corresponding SpaCy language model.

    Parameters
    ----------
    language : str
        Language whose stop words will be removed. Default is 'en'.

    Returns
    -------
    callable
        Function that removes stop words from the tokenized part of the input data.

    Notes
    -----
    This function does not lowercase the tokenized data prior to stopword removal.
    """

    try:
        import spacy

        disable = ["tagger", "parser", "ner"]
        nlp = spacy.load(language, disable=disable)
    except OSError:
        warnings.warn(
            f"SpaCy model {language} not found. Trying to download and install."
        )

        from spacy.cli.download import download

        download(language)
        nlp = spacy.load(language, disable=disable)

    stop_words = nlp.Defaults.stop_words

    def _remove_hook(raw, tokenized):
        tokenized = [
            token for token in nlp.Defaults.stop_words if token not in stop_words
        ]

        return raw, tokenized

    return _remove_hook
