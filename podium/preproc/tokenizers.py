"""
Module contains text tokenizers.
"""

from podium.utils.general_utils import load_spacy_model_or_raise


def get_tokenizer(tokenizer):
    """
    Returns a tokenizer according to the parameters given.

    Parameters
    ----------
    tokenizer : str | callable
        If a callable object is given, it will just be returned.
        Otherwise, a string can be given to create one of the premade
        tokenizers. The string must be of format 'tokenizer' or `tokenizer-args`

        The available premade tokenizers are:
            - 'split' - default str.split(). Custom separator can be provided as
              `split-sep` where `sep` is the separator string.

            - 'spacy' - the spacy tokenizer, using the 'en' language
              model by default . Different language model can be provided as
              'spacy-lang' where `lang` is the language model name (e.g. `spacy-en`).
              If spacy model is used for the first time, an attempt to install it will be
              made. If that fails, user should download it by using command similar
              to the following `python -m spacy download en`.
              More details can be found in spacy documentation https://spacy.io/usage/models.

            - toktok - NLTK's toktok tokenizer. For more details
              see https://www.nltk.org/_modules/nltk/tokenize/toktok.html.

            - moses - Sacremoses's moses tokenizer. For more details
              see https://github.com/alvations/sacremoses.

    Returns
    -------
        The created (or given) tokenizer.

    Raises
    ------
    ImportError
        If the required package for the specified tokenizer is not installed.
    ValueError
        If the given tokenizer is not a callable or a string, or is a
        string that doesn't correspond to any of the supported tokenizers.
    """

    if callable(tokenizer):
        return tokenizer

    if not isinstance(tokenizer, str):
        raise ValueError(
            f"Wrong type passed to `get_tokenizer`. Allowed types are callables "
            f"and strings. The provided type is {type(tokenizer)}"
        )

    tokenizer, *language_or_sep = tokenizer.split("-", 1)
    language_or_sep = language_or_sep[0] if language_or_sep else None

    if tokenizer == "spacy":
        language = language_or_sep if language_or_sep is not None else "en_core_web_sm"
        spacy = load_spacy_model_or_raise(language, disable=["parser", "ner"])

        # closures instead of lambdas because they are serializable
        def spacy_tokenize(string):
            # need to wrap in a function to access .text
            return [token.text for token in spacy.tokenizer(string)]

        return spacy_tokenize

    elif tokenizer == "split":
        sep = language_or_sep

        def _split(string):
            return string.split(sep)

        return _split

    elif tokenizer == "toktok":
        from nltk.tokenize.toktok import ToktokTokenizer

        toktok = ToktokTokenizer()
        return toktok.tokenize

    elif tokenizer == "moses":
        try:
            from sacremoses import MosesTokenizer

            moses_tokenizer = MosesTokenizer()
            return moses_tokenizer.tokenize
        except ImportError:
            print(
                "Please install SacreMoses. "
                "See the docs at https://github.com/alvations/sacremoses "
                "for more information."
            )
            raise
    else:
        raise ValueError(f"Wrong value given for the tokenizer: {tokenizer}")
