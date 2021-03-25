"""
Module contains various pretokenization and posttokenization hooks.
"""
import functools
import re
from enum import Enum, auto
from typing import Callable, Iterable, List, Optional, Pattern, Tuple, Union

from nltk.stem import SnowballStemmer

from podium.utils.general_utils import load_spacy_model_or_raise


__all__ = [
    "remove_stopwords",
    "truecase",
    "KeywordExtractor",
    "MosesNormalizer",
    "NLTKStemmer",
    "RegexReplace",
    "SpacyLemmatizer",
    "TextCleanUp",
]

_LANGUAGES = {
    "ar": "arabic",
    "en": "english",
    "da": "danish",
    "nl": "dutch",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "hu": "hungarian",
    "it": "italian",
    "nb": "norwegian",
    "ro": "romanian",
    "ru": "russian",
    "sv": "swedish",
    "pt": "portuguese",
    "es": "spanish",
}
_LANGUAGES.update({lang: lang for lang in _LANGUAGES.values()})


class HookType(Enum):
    PRETOKENIZE = auto()
    POSTTOKENIZE = auto()


def pretokenize_hook(hook):
    hook.__hook_type__ = HookType.PRETOKENIZE
    return hook


def posttokenize_hook(hook):
    hook.__hook_type__ = HookType.POSTTOKENIZE
    return hook


def as_posttokenize_hook(hook):
    """
    Transforms a pretokenitation hook to a posttokenization hook.

    This function supports only the built-in hooks and raises TypeError for the
    user-provided hooks.
    """
    try:
        hook_type = hook.__hook_type__
    except AttributeError:
        raise TypeError("Only a built-in hook can be transformed")
    else:
        if hook_type == HookType.PRETOKENIZE:

            @posttokenize_hook
            def posttokenize_hook_(raw, tokenized):
                return raw, [hook(token) for token in tokenized]

            return posttokenize_hook_
        else:
            return hook


def truecase(oov: str = "title") -> Callable[[str], str]:
    """
    Returns a pretokenization hook that applies truecasing to the raw textual
    data.

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

    @pretokenize_hook
    def _truecase_hook(raw):
        return truecase_.get_true_case(raw, out_of_vocabulary_token_option=oov)

    return _truecase_hook


def remove_stopwords(
    language: str = "en",
) -> Callable[[str, List[str]], Tuple[str, List[str]]]:
    """
    Returns a posttokenization hook that removes stop words from the tokenized
    textual data. The raw part is left unchanged.

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

    language = "en_core_web_sm" if language == "en" else language
    nlp = load_spacy_model_or_raise(language, disable=["tagger", "parser", "ner"])
    stop_words = nlp.Defaults.stop_words

    @posttokenize_hook
    def _remove_hook(raw, tokenized):
        tokenized = [token for token in tokenized if token not in stop_words]

        return raw, tokenized

    return _remove_hook


@pretokenize_hook
class MosesNormalizer:
    """
    Pretokenization took that normalizes the raw textual data.

    Uses sacremoses.MosesPunctNormalizer to perform normalization.
    """

    def __init__(self, language: str = "en") -> None:
        """
        MosesNormalizer constructor.

        Parameters
        ----------
        language : str
            Language argument for the normalizer. Default: "en".

        Raises
        ------
        ImportError
            If sacremoses is not installed.
        """
        try:
            from sacremoses import MosesPunctNormalizer
        except ImportError:
            print(
                "Problem occured while trying to import sacremoses. "
                "If the library is not installed visit "
                "https://github.com/alvations/sacremoses for more details."
            )
            raise

        self._normalizer = MosesPunctNormalizer(language)

    def __call__(self, raw: str) -> str:
        """
        Applies normalization to the raw textual data.

        Parameters
        ----------
        raw : str
            Raw textual data.

        Returns
        -------
        str
            Normalized textual data.
        """
        return self._normalizer.normalize(raw)


@pretokenize_hook
class RegexReplace:
    """
    Pretokenization hook that applies a sequence of regex substitutions to the
    raw textual data.

    Each substitution corresponds to a 2-tuple consisting of a regex pattern and
    a string that will replace that pattern.
    """

    def __init__(
        self,
        replace_patterns: Iterable[Tuple[Union[Pattern, str], str]],
    ) -> None:
        """
        RegexReplace constructor.

        Parameters
        ----------
        replace_patterns : iterable of tuple(Union[re.Pattern, str], str)
            Iterable of 2-tuples where the first element is either
            a regex pattern or a string and the second element
            is a string that will replace each occurance of the pattern specified as
            the first element.
        """
        self._patterns = [
            (re.compile(pattern), repl) for pattern, repl in replace_patterns
        ]

    def __call__(self, raw: str) -> str:
        """
        Applies a sequence of regex substitutions to the raw textual data.

        Parameters
        ----------
        raw : str
            Raw textual data.

        Returns
        -------
        str
            Resulting textual data after applying the regex substitutions.
        """
        for re_pattern, repl in self._patterns:
            raw = re_pattern.sub(repl, raw)

        return raw


@pretokenize_hook
class TextCleanUp:
    """
    Pretokenization hook that cleans up the raw textual data.

    Additionally, it supports replacement of urls, emails, phone numbers,
    numbers, digits, and currency symbols with arbitrary tokens. During the
    clean up, whitespace is normalized.
    """

    def __init__(
        self,
        language="en",
        fix_unicode: bool = True,
        to_ascii: bool = True,
        remove_line_breaks: bool = False,
        remove_punct: bool = False,
        replace_url: Optional[str] = None,
        replace_email: Optional[str] = None,
        replace_phone_number: Optional[str] = None,
        replace_number: Optional[str] = None,
        replace_digit: Optional[str] = None,
        replace_currency_symbol: Optional[str] = None,
    ) -> None:
        """
        TextCleanUp constructor.

        Parameters
        ----------
        language : str
            Language argument for the text clean up. Default: "en".
        fix_unicode : bool
            Fix various unicode errors. Default: True.
        to_ascii : bool
            Transliterate to closest ASCII representation. Default: True.
        remove_line_breaks : bool
            Fully strip line breaks as opposed to only normalizing them. Default: False.
        remove_punct : bool
            Fully remove punctuation. Default: False.
        replace_url : str, optional
            If defined, token used to replace urls in the input data. Default: None.
        replace_email : str, optional
            If defined, token used to replace emails in the input data. Default: None.
        replace_phone_number : str, optional
            If defined, token used to replace phone numbers in the input data.
            Default: None.
        replace_number : str, optional
            If defined, token used to replace numbers in the input data. Default: None.
        replace_digit : str, optional
            If defined, token used to replace digits in the input data. Default: None.
        replace_currency_symbol : str, optional
            If defined, token used to replace currency symbols in the input data.
            Default: None.

        Raises
        ------
        ValueError
            If the given language is not supported.
        """
        try:
            from cleantext.clean import clean
        except ImportError:
            print(
                "Problem occured while trying to import clean-text. "
                "If the library is not installed visit "
                "https://pypi.org/project/clean-text/ for more details."
            )
            raise

        if language not in {"en", "de"}:
            raise ValueError(f"Language {language} is not supported.")

        kwargs = {
            "lang": language,
            "fix_unicode": fix_unicode,
            "to_ascii": to_ascii,
            "no_line_breaks": remove_line_breaks,
            "no_punct": remove_punct,
            "lower": False,
        }

        def _replace_pattern(pattern_tag, value, kwargs):
            if isinstance(value, str):
                kwargs.update(
                    {
                        "no_" + pattern_tag + "s": True,
                        "replace_with_" + pattern_tag: value,
                    }
                )

        _replace_pattern("url", replace_url, kwargs)
        _replace_pattern("email", replace_email, kwargs)
        _replace_pattern("phone_number", replace_phone_number, kwargs)
        _replace_pattern("number", replace_number, kwargs)
        _replace_pattern("digit", replace_digit, kwargs)
        _replace_pattern("currency_symbol", replace_currency_symbol, kwargs)

        self._cleanup = functools.partial(clean, **kwargs)

    def __call__(self, raw: str) -> str:
        """
        Cleans up the raw textual data.

        Parameters
        ----------
        raw : str
            Raw textual data.

        Returns
        -------
        str
            Cleaned up textual data.
        """
        return self._cleanup(raw)


@posttokenize_hook
class KeywordExtractor:
    """
    Posttokenization hook that extracts keywords from the raw textual data.

    The tokenized data is ignored during this process.
    """

    def __init__(self, algorithm: str, **kwargs) -> None:
        """
        Keyword Extractor constructor.

        Parameters
        ----------
        algorithm : str
            The algorithm used to extract keywords. Supported algorithms: `rake` and `yake`.
        **kwargs: keyword arguments passed to the keyword extraction algorithm.

        Raises
        ------
        ImportError
            If the keyword extraction algorithm is not installed.
        ValueError
            If the specified extraction algorithm is not supported.
        """
        if algorithm == "rake":
            try:
                import rake_nltk
            except ImportError:
                print(
                    "Problem occured while trying to import rake-nltk. "
                    "If the library is not installed visit "
                    "https://csurfer.github.io/rake-nltk/_build/html/index.html "
                    "for more details."
                )
                raise

            extractor = rake_nltk.Rake(**kwargs)
        elif algorithm == "yake":
            try:
                import yake
            except ImportError:
                print(
                    "Problem occured while trying to import yake. "
                    "If the library is not installed visit "
                    "https://github.com/LIAAD/yake for more details."
                )
                raise

            extractor = yake.KeywordExtractor(**kwargs)

        else:
            raise ValueError(
                f"{algorithm} is not supported as keyword extraction algorithm. "
                f"Available algorithms: {['rake', 'yake']}"
            )

        self._algorithm = algorithm
        self._kw_extractor = extractor

    def __call__(self, raw: str, tokenized: List[str]) -> Tuple[str, List[str]]:
        """
        Extracts keywords from the raw data.

        Returns
        -------
        tuple(str, list of str)
            2-tuple where the first element is left unchanged and the second
            elements contains extracted keywords.
        """
        if self._algorithm == "rake":
            self._kw_extractor.extract_keywords_from_text(raw)
            keywords = self._kw_extractor.get_ranked_phrases()
        elif self._algorithm == "yake":
            keywords = [kw for kw, _ in self._kw_extractor.extract_keywords(raw)]
        return raw, keywords


@posttokenize_hook
class NLTKStemmer:
    """
    Posttokenization hook that applies stemming to the tokenized textual data.

    Uses nltk.stem.SnowballStemmer to perform stemming.
    """

    def __init__(self, language: str = "en", ignore_stopwords: bool = False) -> None:
        """
        NLTKStemmer constructor.

        Parameters
        ----------
        language : str
            The language argument for the stemmer. Default: "en".
            For the list of supported language,
            see: https://www.nltk.org/api/nltk.stem.html.
        ignore_stopwords : bool
            If True, stemming is not applied to stopwords. Default: False.

        Raises
        ------
        ValueError
            If the given language is not supported.
        """
        if language in _LANGUAGES:
            language = _LANGUAGES[language]
        elif language != "porter":
            raise ValueError(f"Language {language} is not supported.")

        self._stemmer = SnowballStemmer(language, ignore_stopwords)

    def __call__(self, raw: str, tokenized: List[str]) -> Tuple[str, List[str]]:
        """
        Stemms the tokenized textual data. The raw part is left unchanged.

        Returns
        -------
        tuple(str, list of str)
            2-tuple where the first element is left unchanged and the second
            elements contains stemmed tokens.
        """
        return raw, [self._stemmer.stem(token) for token in tokenized]


@posttokenize_hook
class SpacyLemmatizer:
    """
    Posttokenization hook that applies SpaCy Lemmatizer to the tokenized textual
    data.

    If the language model is not installed, an attempt is made to install it.
    """

    def __init__(self, language: str = "en", mode: str = "lookup") -> None:
        """
        SpacyLemmatizer constructor.

        Parameters
        ----------
        language : str
            Language argument for the lemmatizer.
            For the list of supported languages,
            see https://spacy.io/usage/models#languages.
            Default: "en".
        mode : str
            The lemmatizer mode. By default, the following modes are available:
            "lookup" and "rule". Default: "lookup".
        """

        language = "en_core_web_sm" if language == "en" else language
        nlp = load_spacy_model_or_raise(language, disable=["parser", "ner"])

        try:
            # SpaCy<3.0
            from spacy.lemmatizer import Lemmatizer

            is_spacy_old = True
        except ImportError:
            # SpaCy>=3.0
            from spacy.pipeline import Lemmatizer
            from spacy.tokens import Doc

            is_spacy_old = False

        if is_spacy_old:
            lemmatizer = Lemmatizer(nlp.vocab.lookups)

            if mode == "lookup":
                lemmatizer.lookups.remove_table("lemma_rules")
                lemmatizer.lookups.remove_table("lemma_index")
                lemmatizer.lookups.remove_table("lemma_exc")
            else:
                lemmatizer.lookups.remove_table("lemma_lookup")

            def lemmatize(tokenized):
                return [lemmatizer.lookup(token) for token in tokenized]

        else:
            lemmatizer = Lemmatizer(nlp.vocab, None, mode=mode)
            try:
                lemmatizer.initialize()
            except ValueError as err:
                raise ValueError(
                    "SpaCy lookups data is missing. "
                    "Visit https://spacy.io/usage/models"
                    "for more information on how to install it."
                ) from err

            def tokenizer(text: List[str]) -> Doc:
                return Doc(nlp.vocab, text)

            nlp.tokenizer = tokenizer

            def lemmatize(tokenized):
                return [token.lemma_ for token in lemmatizer(nlp(tokenized))]

        self._lemmatize = lemmatize

    def __call__(self, raw: str, tokenized: List[str]) -> Tuple[str, List[str]]:
        """
        Applies lemmatization to the tokenized textual data. The raw part is
        left unchanged.

        Returns
        -------
        tuple(str, list of str)
            2-tuple where the first element is left unchanged and the second
            elements contains lemmatized tokens.
        """
        return raw, self._lemmatize(tokenized)
