"""Module contains text tokenizers."""
import logging

_LOGGER = logging.getLogger(__name__)


def get_tokenizer(tokenizer, language='en'):
    """
    Returns a tokenizer according to the parameters given.

    Parameters
    ----------
    tokenizer : str | callable
        If a callable object is given, it will just be returned.
        Otherwise, a string can be given to create one of the premade
        tokenizers.
        The available premade tokenizers are:
            - 'split' - default str.split()
            - 'spacy' - the spacy tokenizer, using the 'en' language
            model by default (unless the user provides a different
            'language' parameter). If spacy model is used for the first time
            user should download it by using command similar to the following
            `python -m spacy download en`. More details can be found in spacy
            documentation https://spacy.io/usage/models

    language : str
        The language argument for the tokenizer (if necessary, e. g. for
        spacy). Default is 'en'.

    Returns
    -------
        The created (or given) tokenizer.

    Raises
    ------
    ValueError
        If the given tokenizer is not a callable or a string, or is a
        string that doesn't correspond to any of the premade tokenizers.
    """
    # Add every new tokenizer to this "factory" method
    if callable(tokenizer):
        # if arg is already a function, just return it
        return tokenizer

    elif tokenizer == 'spacy':
        try:
            import spacy
            spacy_tokenizer = spacy.load(language, disable=['parser', 'ner'])

            # closures instead of lambdas because they are serializable
            def spacy_tokenize(string):
                # need to wrap in a function to access .text
                return [token.text for token in
                        spacy_tokenizer.tokenizer(string)]

            return spacy_tokenize
        except (ImportError, OSError) as ex:
            error_msg = f'Please install SpaCy and the SpaCy {language} '\
                        f'tokenizer. See the docs at https://spacy.io for '\
                        f'more information.'
            _LOGGER.error(error_msg)
            raise ex

    elif tokenizer == "split":
        return str.split

    else:
        error_msg = f"Wrong value given for the tokenizer: "\
                    f"{tokenizer}"
        _LOGGER.error(error_msg)
        raise ValueError(error_msg)
