"""
Module contains various pretokenization and posttokenization hooks.
"""
import functools
import re
from typing import List, Optional, Pattern, Sequence, Tuple, Union

from nltk.stem import SnowballStemmer

from podium.utils.general_utils import load_spacy_model_or_raise


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


class _DualHook:
    """
    A mixin class which allows a hook class to be cast as both a pretokenization
    and posttokenization hook via the `as_pretokenization` constructor argument.
    """

    def __init__(self, as_pretokenization=True):
        """
        Allows subclasses to be cast as pretokenization or posttokenization
        hooks.

        Parameters
        ----------
        as_pretokenization : bool
            A boolean flag which indicates whether this hook should be used during
            as_pretokenization (if True) or posttokenization (if False),
        """
        self.as_pretokenization = as_pretokenization

    def apply(self, string):
        """
        Applies the hook to a string input.

        Should be overrided by implementing methods.
        """
        return string

    def run_as_pretokenization(self, raw):
        """
        Apply the hook to a raw string input.
        """
        return self.apply(raw)

    def run_as_posttokenization(self, raw, tokenized):
        """
        Apply the hook to a tokenized sequence.
        """
        return raw, [self.apply(token) for token in tokenized]

    def __call__(self, *args):
        """
        Apply the hook to either a tokenized sequence or a raw string input.
        """
        if self.as_pretokenization:
            return self.run_as_pretokenization(*args)
        else:
            return self.run_as_posttokenization(*args)


class MosesNormalizer(_DualHook):
    """
    Pretokenization took that normalizes the raw textual data.

    Uses sacremoses.MosesPunctNormalizer to perform normalization.
    """

    def __init__(self, language: str = "en", as_pretokenization: bool = True) -> None:
        """
        MosesNormalizer constructor.

        Parameters
        ----------
        language : str
            Language argument for the normalizer. Default: "en".
        as_pretokenization : bool
            A boolean flag which indicates whether this hook should be used during
            as_pretokenization (if True) or posttokenization (if False),

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

        super().__init__(as_pretokenization)
        self._normalizer = MosesPunctNormalizer(language)

    def apply(self, string: str) -> str:
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
        return self._normalizer.normalize(string)


class RegexReplace(_DualHook):
    """
    Pretokenization hook that applies a sequence of regex substitutions to the
    raw textual data.

    Each substitution corresponds to a 2-tuple consisting of a regex pattern and
    a string that will replace that pattern.
    """

    def __init__(
        self,
        replace_patterns: Sequence[Tuple[Union[Pattern, str], str]],
        as_pretokenization: bool = True,
    ) -> None:
        """
        RegexReplace constructor.

        Parameters
        ----------
        replace_patterns : sequence of tuple(Union[re.Pattern, str], str)
            Iterable of 2-tuples where the first element is either
            a regex pattern or a string and the second element
            is a string that will replace each occurance of the pattern specified as
            the first element.
        as_pretokenization : bool
            A boolean flag which indicates whether this hook should be used during
            as_pretokenization (if True) or posttokenization (if False),
        """
        super().__init__(as_pretokenization)
        self._patterns = [
            (re.compile(pattern), repl) for pattern, repl in replace_patterns
        ]

    def apply(self, raw: str) -> str:
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
                    "for more information on how to install."
                ) from err

            def tokenizer(text: List[str]) -> Doc:
                return Doc(nlp.vocab, text)

            nlp.tokenizer = tokenizer

            def lemmatize(tokenized):
                return [token.lemma_ for token in lemmatizer(nlp(tokenized))]

        self._lemmatize = lemmatize
        # nlp.tokenizer = tokenizer
        # self._str2token = nlp
        # self._lemmatizer = Lemmatizer(nlp.vocab, None, mode=mode)

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
