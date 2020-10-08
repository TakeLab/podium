"""Module contains text tokenizers."""
import logging
from podium.util import log_and_raise_error

_LOGGER = logging.getLogger(__name__)

_registered_tokenizer_factories = {}


def tokenizer_factory(tokenizer_name):

    def register_tokenizer(tokenizer_factory_method):
        _registered_tokenizer_factories[tokenizer_name] = tokenizer_factory_method
        return tokenizer_factory_method

    return register_tokenizer


def list_tokenizers():
    return list(_registered_tokenizer_factories.keys())


def get_tokenizer(tokenizer):
    """Returns a tokenizer according to the parameters given.

    Parameters
    ----------
    tokenizer : str | callable
        If a callable object is given, it will just be returned.
        Otherwise, a string can be given to create one of the premade
        tokenizers. The string must be of format 'tokenizer' or `tokenizer-args`

        The available premade tokenizers are:
            - 'split' - default str.split()

            - 'spacy' - the spacy tokenizer, using the 'en' language
              model by default (unless the user provides a different
              language trough args). If spacy model is used for the first time
              user should download it by using command similar to the following
              `python -m spacy download en`. More details can be found in spacy
              documentation https://spacy.io/usage/models

    Returns
    -------
        The created (or given) tokenizer.

    Raises
    ------
    ValueError
        If the given tokenizer is not a callable or a string, or is a
        string that doesn't correspond to any of the supported tokenizers.
    """

    # Add every new tokenizer to this "factory" method
    if callable(tokenizer):
        # if arg is already a function, just return it
        return tokenizer

    elif isinstance(tokenizer, str):
        tokenizer_split = tokenizer.split('-', 1)
        if len(tokenizer_split) == 1:
            tokenizer_name, tokenizer_args = tokenizer_split[0], None
        else:
            tokenizer_name, tokenizer_args = tokenizer_split

        if tokenizer_name in _registered_tokenizer_factories:
            return _registered_tokenizer_factories[tokenizer_name](tokenizer_args)

        else:
            error_msg = f"Tokenizer {tokenizer_name} not registered in podium."
            log_and_raise_error(ValueError, _LOGGER, error_msg)

    else:  # if tokenizer not found
        error_msg = f"Wrong value given for the tokenizer: {tokenizer}"
        log_and_raise_error(ValueError, _LOGGER, error_msg)


@tokenizer_factory('spacy')
def _get_spacy_tokenizer(language):
    import spacy
    if language is None or language == "":
        warning_msg = f"No language was provided for the spacy tokenizer. " \
                      f"Please provide a language in the tokenizer definition " \
                      f"as 'spacy-language'. Defaulting to 'en'."
        _LOGGER.warning(warning_msg)
        language = 'en'

    disable = ["parser", "ner"]
    try:
        spacy_tokenizer = spacy.load(language, disable=disable)
    except OSError:
        _LOGGER.warning("SpaCy model {} not found."
                        "Trying to download and install."
                        .format(language))

        from spacy.cli.download import download
        download(language)
        spacy_tokenizer = spacy.load(language, disable=disable)

    def spacy_tokenizer(string):
        # need to wrap in a function to access .text
        return [token.text for token in
                spacy_tokenizer.tokenizer(string)]

    return spacy_tokenizer


@tokenizer_factory('split')
def _get_split_tokenizer(arg):
    def split(s, sep=arg):
        return s.split(sep)
    return split
