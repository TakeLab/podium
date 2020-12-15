"""Module contains various pretokenization and posttokenization hooks."""
import functools
import re
import warnings
from typing import List, Optional, Pattern, Sequence, Tuple, Union

import spacy
from nltk.stem import SnowballStemmer
from spacy.lemmatizer import Lemmatizer


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


class MosesNormalizer:
    """Pretokenization took that normalizes the raw textual data.

    Uses sacremoses.MosesPunctNormalizer to perform normalization.
    """

    def __init__(self, language: str = "en") -> None:
        """MosesNormalizer constructor.

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
        """Applies normalization to the raw textual data.

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


class RegexReplace:
    """Pretokenization hook that applies a sequence of regex substitutions to the raw
    textual data. Each substitution corresponds to a 2-tuple consisting of
    a regex pattern and a string that will replace that pattern."""

    def __init__(
        self,
        replace_patterns: Sequence[Tuple[Union[Pattern, str], str]],
    ) -> None:
        """RegexReplace constructor.

        Parameters
        ----------
        replace_patterns : sequence of tuple(Union[re.Pattern, str], str)
            Iterable of 2-tuples where the first element is either
            a regex pattern or a string and the second element
            is a string that will replace each occurance of the pattern specified as
            the first element.
        """
        self._patterns = [
            (re.compile(pattern), repl) for pattern, repl in replace_patterns
        ]

    def __call__(self, raw: str) -> str:
        """Applies a sequence of regex substitutions to the raw
        textual data.

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


class TextCleanUp:
    """Pretokenization hook that cleans up the raw textual data. Additionally, it supports
    replacement of urls, emails, phone numbers, numbers, digits, and currency symbols
    with arbitrary tokens. During the clean up, whitespace is normalized.
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
        """TextCleanUp constructor.

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
            from cleantext import clean
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
        """Cleans up the raw textual data.

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


class NLTKStemmer:
    """Posttokenization hook that applies stemming to the tokenized textual data.

    Uses nltk.stem.SnowballStemmer to perform stemming.
    """

    def __init__(self, language: str = "en", ignore_stopwords: bool = False) -> None:
        """NLTKStemmer constructor.

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
        """Stemms the tokenized textual data. The raw part is left unchanged.

        Returns
        -------
        tuple(str, list of str)
            2-tuple where the first element is left unchanged and the second
            elements contains stemmed tokens.
        """
        return raw, [self._stemmer.stem(token) for token in tokenized]


class SpacyLemmatizer:
    """Posttokenization hook that applies SpaCy Lemmatizer to the tokenized textual data.
    If the language model is not installed, an attempt is made to install it.
    """

    def __init__(self, language: str = "en") -> None:
        """SpacyLemmatizer constructor.

        Parameters
        ----------
        language : str
            Language argument for the lemmatizer.
            For the list of supported languages,
            see https://spacy.io/usage/models#languages.
            Default: "en".
        """
        try:
            disable = ["tagger", "parser", "ner"]
            nlp = spacy.load(language, disable=disable)
        except OSError:
            warnings.warn(
                f"SpaCy model {language} not found. Trying to download and install."
            )

            from spacy.cli.download import download

            download(language)
            nlp = spacy.load(language, disable=disable)

        self._lemmatizer = Lemmatizer(nlp.vocab.lookups)

    def __call__(self, raw: str, tokenized: List[str]) -> Tuple[str, List[str]]:
        """Applies lemmatization to the tokenized textual data.
        The raw part is left unchanged.

        Returns
        -------
        tuple(str, list of str)
            2-tuple where the first element is left unchanged and the second
            elements contains lemmatized tokens.
        """
        return raw, [self._lemmatizer.lookup(token) for token in tokenized]
