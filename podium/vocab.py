"""
Module contains classes related to the vocabulary.
"""
import itertools
import warnings
from collections import Counter
from typing import Any
from typing import Counter as Counter_
from typing import (
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np

from podium.utils.general_utils import repr_type_and_attrs


T = TypeVar("T", bound=Hashable)


def _unique(values: Iterable[T]) -> Iterator[T]:
    """
    Generator that iterates over the first occurrence of every value in
    `values`, preserving original order.

    Parameters
    ----------
    values: Iterable
        Iterable of values

    Yields
    -------
        the first occurrence of every value in values, preserving order.
    """
    seen = set()
    for element in values:
        if element in seen:
            continue
        seen.add(element)
        yield element


class Special(str):
    """
    Base class for a special token.

    Every special token is a subclass of string (this way one can) easily modify
    the concrete string representation of the special. The functionality of the
    special token, which acts the same as a post-tokenization hook should be
    implemented in the `apply` instance method for each subclass. We ensure that
    each special token will be present in the Vocab.
    """

    token: Optional[str] = None

    def __new__(cls, token: Optional[str] = None):
        """
        Provides default value initialization for subclasses.

        If creating a new instance without a string argument, the `token` class
        attribute must be set in the subclass implementation.
        """

        if token is None:
            token = cls.token

        if token is None:
            raise ValueError(
                "When initializing a special token without argument"
                f" the {cls.__name__}.token attribute must be set."
            )

        return super(Special, cls).__new__(cls, token)

    def __hash__(self) -> int:
        """
        Overrides hash.

        Check docs of `__eq__` for motivation.
        """
        return hash(self.__class__)

    def __eq__(self, other: Any) -> bool:
        """
        Check equals via class instead of value.

        The motivation behind this is that we want to be able to match the
        special token by class and not by value, as it is the type of the
        special token that determines its functionality. This way we allow for
        the concrete string representation of the special to be easily changed,
        while retaining simple existence checks for vocab functionality.
        """
        return self.__class__ == other.__class__

    def apply(self, sequence: Sequence[str]) -> Sequence[str]:
        """
        Apply (insert) the special token in the adequate place in the sequence.

        By default, returns the unchanged sequence.
        """
        return sequence


class BOS(Special):
    """
    The beginning-of-sequence special token.
    """

    token = "<BOS>"

    def apply(self, sequence):
        """
        Apply the BOS token, adding it to the start of the sequence.
        """
        return [self] + sequence


class EOS(Special):
    """
    The end-of-sequence special token.
    """

    token = "<EOS>"

    def apply(self, sequence):
        """
        Apply the EOS token, adding it to the end of the sequence.
        """
        return sequence + [self]


#################
# Core specials #
#################


class UNK(Special):
    """
    The unknown core special token.

    Functionality handled by Vocab.
    """

    token = "<UNK>"


class PAD(Special):
    """
    The padding core special token.

    Functionality handled by Vocab.
    """

    token = "<PAD>"


class Vocab:
    """
    Class for storing vocabulary. It supports frequency counting and size
    limiting.

    Attributes
    ----------
    is_finalized : bool
        true if the vocab is finalized, false otherwise
    itos : list
        list of words
    stoi : dict
        mapping from word string to index
    """

    _unk: Special = UNK()
    _pad: Special = PAD()

    def __init__(
        self,
        max_size: Optional[int] = None,
        min_freq=1,
        specials: Optional[Union[Special, Iterable[Special]]] = (UNK(), PAD()),
        keep_freqs: bool = False,
        eager: bool = False,
    ) -> None:
        """
        Vocab constructor. Specials are first in the vocabulary.

        Parameters
        ----------
        max_size : int
            maximal vocab size
        min_freq : int
            words with frequency lower than this will be removed
        specials : Special | Tuple(Special) | None
            collection of special symbols.
            Can be None.
        keep_freqs : bool
            if `True` word frequencies will be saved for later use on
            the finalization
        eager : bool
            if `True` the frequencies will be built immediately upon
            dataset loading. The main effect of this argument if set
            to `True` is that the frequencies of the vocabulary will
            be built based on all datasets that use this vocabulary,
            while if set to `False`, the vocabulary will be built
            by iterating again over the datasets passed as argument
            to the `finalize_fields` function. If you are using multiple
            datasets and wish to manually control on which subset of
            dataset splits the vocab is built on, eager should be False.
            If you are using one or multiple large datasets and/or want to
            build the vocabulary on all of the splits, eager should be
            set to True for performance optimization (one loop over the
            datasets instead of two).
        """
        self._max_size = max_size
        self._min_freq = min_freq

        if specials is None:
            specials = ()
        elif isinstance(specials, Special):
            specials = (specials,)

        self._specials = tuple(specials)

        # Apply uniqueness check
        if len(self.specials) > len(set(self.specials)):
            raise ValueError("Specials may not contain multiple instances of same type.")

        self._itos = list(self.specials)
        self._stoi = {k: v for v, k in enumerate(self.itos)}

        self._keep_freqs = keep_freqs
        self._eager = eager
        self._is_finalized = False  # flag to know if we're ready to numericalize

        self._freqs: Optional[Counter_[str]] = Counter()

    @property
    def eager(self) -> bool:
        return self._eager

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    @property
    def specials(self) -> Tuple:
        return self._specials

    @property
    def itos(self) -> List[str]:
        return self._itos

    @property
    def stoi(self) -> Dict[str, int]:
        return self._stoi

    @classmethod
    def from_itos(cls, itos: Iterable[str]):
        """
        Method constructs a vocab from a predefined index-to-string mapping.

        Parameters
        ----------
            itos : list | tuple
                The index-to-string mapping for tokens in the vocabulary
        """
        specials = tuple(token for token in itos if isinstance(token, Special))

        vocab = cls(specials=specials)
        vocab._itos = list(itos)
        vocab._stoi = {v: k for k, v in enumerate(itos)}
        vocab._is_finalized = True

        return vocab

    @classmethod
    def from_stoi(cls, stoi: Dict[str, int]):
        """
        Method constructs a vocab from a predefined index-to-string mapping.

        Parameters
        ----------
            stoi : dict
                The string-to-index mapping for the vocabulary
        """
        specials = tuple(token for token in stoi.keys() if isinstance(token, Special))

        vocab = cls(specials=specials)
        vocab._stoi = stoi
        vocab_max_index = max(stoi.values())
        itos: List[str] = [None] * (vocab_max_index + 1)
        for token, index in stoi.items():
            itos[index] = token
        vocab._itos = itos
        vocab._is_finalized = True

        return vocab

    def get_freqs(self) -> Counter_[str]:
        """
        Method obtains vocabulary frequencies.
        Returns
        -------
        freq : Counter
            mapping frequency for every word
        Raises
        ------
        RuntimeError
            If the user stated that he doesn't want to keep frequencies
            and the vocab is finalized.
        """
        if self.is_finalized and not self._keep_freqs:
            raise RuntimeError(
                "User specified that frequencies aren't kept in "
                "vocabulary but the get_freqs method is called."
            )
        return self._freqs

    def get_padding_index(self) -> int:
        """
        Method returns padding symbol index.

        Returns
        -------
        pad_symbol_index : int
            padding symbol index in the vocabulary

        Raises
        ------
        ValueError
            If the padding symbol is not present in the vocabulary.
        """
        if Vocab._pad not in self.stoi:
            raise ValueError("Padding symbol is not in the vocabulary.")
        return self.stoi[Vocab._pad]

    def __iadd__(self, values: Union["Vocab", Iterable[str]]) -> "Vocab":
        """
        Adds additional values or another Vocab to this Vocab.

        Parameters
        ----------
        values : Iterable or Vocab
            Values to be added to this Vocab.
            If Vocab, all of the token frequencies and specials from that Vocab will be
            added to this Vocab. Wheen adding two Vocabs with a different string values
            for a special token, only the special token instance with the valuefrom the
            LHS operand will be used.

            If Iterable, all of the tokens from the Iterable will be added to this Vocab,
            increasing the frequencies of those tokens.

        Returns
        -------
        vocab : Vocab
            Returns current Vocab instance to enable chaining

        Raises
        ------
        RuntimeError
            If the current vocab is finalized, if 'values' is a string or if the
            RHS Vocab doesn't contain token frequencies.

        TypeError
            If the values cannot be iterated over.
        """
        if self.is_finalized:
            raise RuntimeError("Once finalized, vocabulary cannot be changed.")

        if isinstance(values, str):
            raise TypeError(
                "Vocabulary doesn't support adding a string. "
                "If you need single word added to vocab,"
                " you should wrap it to an iterable."
            )
            # if it is a string characters of a string will be added to counter
            # instead of whole string

        if isinstance(values, Vocab):
            other_vocab = values

            if other_vocab._freqs is None:
                raise RuntimeError(
                    "Error while adding Vocabs inplace. "
                    "RHS Vocab doesn't have word frequencies stored."
                    " Try adding a non-finalized vocab or a Vocab with "
                    "`keep_freqs` enabled."
                )

            # unique is used instead of set to somewhat preserve ordering
            self._specials = tuple(
                _unique(itertools.chain(self.specials, other_vocab.specials))
            )
            self._itos = list(self.specials)
            self._freqs += other_vocab._freqs  # add freqs to this instance

        else:
            try:
                self._freqs.update(
                    value for value in values if value not in self.specials
                )
            except TypeError:
                raise TypeError("Vocab supports only adding another Vocab or iterable.")
        return self

    def __add__(self, values: Union["Vocab", Iterable[str]]) -> "Vocab":
        """
        Method allows a vocabulary to be added to current vocabulary or that a
        set of values is added to the vocabulary.

        If max_size if None for any of the two Vocabs, the max_size of the resulting Vocab
        will also be None. If they are both defined, the max_size of the resulting Vocab
        will be the sum of max_sizes.

        Parameters
        ----------
        values : Iterable or Vocab
            If Vocab, a new Vocab will be created containing all of the special symbols
            and tokens from both Vocabs. Wheen adding two Vocabs with a different string
            values for a special token, only the special token instance with the value
            from the first operand will be used.
            If Iterable, a new Vocab will be returned containing a copy of this Vocab
            with the iterables' tokens added.

        Returns
        -------
        Vocab
            Returns a new Vocab

        Raises
        ------
        RuntimeError
            If this vocab is finalized and values are tried to be added, or
            if both Vocabs are not either both finalized or not finalized.
        """
        if isinstance(values, Vocab):
            other_vocab = values
            specials = tuple(
                _unique(itertools.chain(self.specials, other_vocab.specials))
            )

            if self._max_size is None or other_vocab._max_size is None:
                max_size = None
            else:
                max_size = self._max_size + other_vocab._max_size

            new_vocab = Vocab(
                specials=specials,
                max_size=max_size,
                min_freq=min(self._min_freq, other_vocab._min_freq),
                keep_freqs=self._keep_freqs or other_vocab._keep_freqs,
            )

            if (
                self.is_finalized
                and other_vocab.is_finalized
                and self._keep_freqs
                and other_vocab._keep_freqs
                or not self.is_finalized
                and not other_vocab.is_finalized
            ):
                # If both have _freqs add them together
                new_freqs = self._freqs + other_vocab._freqs
                new_vocab._freqs = new_freqs
                if self.is_finalized:
                    new_vocab.finalize()
                return new_vocab

            elif self.is_finalized and other_vocab.is_finalized:
                new_vocab += self.itos
                new_vocab += other_vocab.itos

                new_vocab.finalize()
                return new_vocab

            else:
                raise RuntimeError(
                    "Vocab addition error. When adding up two Vocabs "
                    "both must be either finalized or not finalized."
                )
        else:
            new_vocab = Vocab(
                specials=self.specials,
                max_size=self._max_size,
                min_freq=self._min_freq,
                keep_freqs=self._keep_freqs,
            )
            new_vocab += self
            new_vocab += values
            if self.is_finalized:
                new_vocab.finalize()
            return new_vocab

    def finalize(self) -> None:
        """
        Method finalizes vocab building. It also releases frequency counter if
        user set not to keep them.

        Raises
        ------
        RuntimeError
            If the vocab is already finalized.
        """
        if self.is_finalized:
            warnings.warn(
                "Vocabulary is finalized already. "
                "This should be used only if multiple fields "
                "use same vocabulary."
            )
            return

        # construct stoi and itos, sort by frequency
        words_and_freqs = sorted(
            self._freqs.items(), key=lambda tup: tup[1], reverse=True
        )
        if self._max_size is None:
            self._max_size = len(words_and_freqs) + len(self.itos)
            # vocab + specials
        for word, freq in words_and_freqs:
            if freq < self._min_freq or len(self.itos) >= self._max_size:
                break
            self.itos.append(word)
            self.stoi[word] = len(self.stoi)

        if not self._keep_freqs:
            del self._freqs  # release memory
        self._is_finalized = True

    def numericalize(self, data: Union[str, Iterable[str]]) -> np.ndarray:
        """
        Method numericalizes given tokens.

        Parameters
        ----------
        data : str | iter(str)
            a single token or iterable collection of tokens

        Returns
        -------
        numericalized_vector : array-like
            numpy array of numericalized tokens

        Raises
        ------
        RuntimeError
            If the vocabulary is not finalized.
        """
        if not self.is_finalized:
            raise RuntimeError(
                "Cannot numericalize if the vocabulary has not been "
                "finalized because itos and stoi are not yet built."
            )

        if isinstance(data, str):
            # Wrap string into list
            data = [data]

        if Vocab._unk in self.stoi:
            # If UNK is in the vocabulary, substitute unknown words with its value
            unk_token = self.stoi[Vocab._unk]
            return np.array(
                [self.stoi[token] if token in self.stoi else unk_token for token in data]
            )
        else:
            # If UNK is not in the vocabulary we filter out unknown words
            return np.array([self.stoi[token] for token in data if token in self.stoi])

    def reverse_numericalize(
        self, numericalized_data: Iterable[int], include_unk=False
    ) -> List[str]:
        """
        Transforms an iterable containing numericalized data into a list of
        tokens. The tokens are read from this Vocab's itos and no additional
        processing is done.

        Parameters
        ----------
        numericalized_data: Iterable
            data to be reverse numericalized

        Returns
        -------
        list
            a list of tokens

        Raises
        ------
        RuntimeError
            If the vocabulary is not finalized.
        """
        if not self.is_finalized:
            raise RuntimeError(
                "Cannot reverse numericalize if the vocabulary has not been "
                "finalized because itos and stoi are not yet built."
            )

        if include_unk and Vocab._unk not in self.specials:
            raise ValueError(
                "`inluce_unk` is set to True but vocab doesn't have the unknown special token."
            )

        if include_unk:
            return [
                self.itos[i] for i in numericalized_data if self.itos[i] != Vocab._unk
            ]
        else:
            return [self.itos[i] for i in numericalized_data]

    def __len__(self) -> int:
        """
        Method calculates vocab lengths including special symbols.

        Returns
        -------
        length : int
            vocab size including special symbols
        """
        if self.is_finalized:
            return len(self.itos)
        return len(self._freqs)

    def __eq__(self, other: Any) -> bool:
        """
        Two vocabs are same if they have same finalization status, their stoi
        and itos mappings are same and their frequency counters are same.

        Parameters
        ----------
        other : object
            object for which we want to knwo equality propertiy

        Returns
        -------
        equal : bool
            true if two vocabs are same, false otherwise
        """

        def _get_freqs_or_none(vocab):
            try:
                return vocab._freqs
            except AttributeError:
                return None

        if self is other:
            return True
        if not isinstance(other, Vocab):
            return False
        if self.is_finalized != other.is_finalized:
            return False
        if _get_freqs_or_none(self) != _get_freqs_or_none(other):
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.specials != other.specials:
            return False
        return True

    def __hash__(self) -> int:
        return hash((self.is_finalized, self._freqs, self.stoi, self.itos, self.specials))

    def __iter__(self) -> Iterable[str]:
        """
        Method returns iterator over vocabulary, if the vocabulary is not
        finalized iteration is done over frequency counter and special symbols
        are not included, otherwise it is performed on itos and special symbols
        are included.

        Returns
        -------
        iter
            iterator over vocab tokens
        """
        if not self.is_finalized:
            return iter(self._freqs.keys())
        return iter(self.itos)

    def __repr__(self) -> str:
        attrs = {
            "specials": self.specials,
            "eager": self.eager,
            "is_finalized": self.is_finalized,
            "size": len(self),
        }
        return repr_type_and_attrs(self, attrs)

    @overload
    def __getitem__(self, idx_or_token: int) -> int:
        ...

    @overload
    def __getitem__(self, idx_or_token: str) -> str:
        ...

    def __getitem__(self, idx_or_token: Union[int, str]) -> Union[int, str]:
        """
        Returns the token index of the passed token. If the passed token has no
        index, UNK token index is returned. Otherwise, an exception is raised.

        Parameters
        ----------
        token: str
            token whose index is to be returned.

        Returns
        -------
        int
            stoi index of the token.

        Raises
        ------
        KeyError
            If the passed token has no index and vocab has no UNK special token.
        """

        if not self.is_finalized:
            raise RuntimeError(
                "`Vocab.itos` and `Vocab.stoi` are not yet built. Please finalize the vocab to build them."
            )

        if isinstance(idx_or_token, int):
            return self.itos[idx_or_token]
        else:
            return self.stoi[idx_or_token]
