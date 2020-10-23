"""Module contains classes related to the vocabulary."""
import logging
from collections import Counter
from enum import Enum
from itertools import chain
from typing import Iterable, Union

import numpy as np


_LOGGER = logging.getLogger(__name__)


def unique(values: Iterable):
    """Generator that iterates over the first occurrence of every value in values,
    preserving original order.

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


class SpecialVocabSymbols(Enum):
    """Class for special vocabular symbols

    Attributes
    ----------
    UNK : str
        Tag for unknown word
    PAD : str
        TAG for padding symbol
    """

    UNK = "<unk>"
    PAD = "<pad>"


class Vocab:
    """Class for storing vocabulary. It supports frequency counting and size
    limiting.

    Attributes
    ----------
    finalized : bool
        true if the vocab is finalized, false otherwise
    itos : list
        list of words
    stoi : dict
        mapping from word string to index
    """

    def __init__(
        self,
        max_size=None,
        min_freq=1,
        specials=(SpecialVocabSymbols.UNK, SpecialVocabSymbols.PAD),
        keep_freqs=False,
        eager=True,
        filter_unk=False
    ):
        """Vocab constructor. Specials are first in the vocabulary.

        Parameters
        ----------
        max_size : int
            maximal vocab size
        min_freq : int
            words with frequency lower than this will be removed
        specials : iter(SpecialVocabSymbols) | None
            collection of special symbols.
            Can be None.
        keep_freqs : bool
            if true word frequencies will be saved for later use on
            the finalization
        eager : bool
            flag indicating whether the Vocab should be built eagerly
        filter_unk : bool
            flag indicating whether unknown tokens should be kept
            when performing numericalization or filtered out.
            By default, the unknown tokens are kept.
        """
        self._freqs = Counter()
        self._keep_freqs = keep_freqs
        self._min_freq = min_freq

        self.specials = () if specials is None else specials
        if not isinstance(self.specials, (tuple, list)):
            self.specials = (self.specials,)
        self._has_specials = len(self.specials) > 0

        self.itos = list(self.specials)
        self._default_unk_index = self._init_default_unk_index(self.specials)
        self.stoi = {k: v for v, k in enumerate(self.itos)}

        self._max_size = max_size
        self.eager = eager
        self.filter_unk = filter_unk
        self.finalized = False  # flag to know if we're ready to numericalize
        _LOGGER.debug("Vocabulary has been created and initialized.")

    @staticmethod
    def _init_default_unk_index(specials):
        """Method computes index of default unknown symbol in given collection.

        Parameters
        ----------
        specials : iter(SpecialVocabSymbols)
            collection of special vocab symbols

        Returns
        -------
        index : int or None
            index of default unkwnown symbol or None if it doesn't exist
        """
        ind = 0
        for spec in specials:
            if spec == SpecialVocabSymbols.UNK:
                return ind
            ind += 1
        return None

    def get_freqs(self):
        """Method obtains vocabulary frequencies.

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
        if self.finalized and not self._keep_freqs:
            raise RuntimeError(
                "User specified that frequencies aren't kept in "
                "vocabulary but the get_freqs method is called."
            )
        return self._freqs

    @property
    def unk_index(self):
        """Method returns default unknown symbol index.

        Returns
        -------
        index: int
            index of default unknown symbol

        Raises
        ------
        ValueError
            If unknown symbol is not present in the vocab.
        """
        if self._default_unk_index is None:
            raise ValueError(
                "Unknown symbol is not present in the vocab but "
                "the user queried a word that isn't in the vocab."
            )
        return self._default_unk_index

    @property
    def padding_index(self):
        """Method returns padding symbol index.

        Returns
        -------
        pad_symbol_index : int
            padding symbol index in the vocabulary

        Raises
        ------
        ValueError
            If the padding symbol is not present in the vocabulary.
        """
        if SpecialVocabSymbols.PAD not in self.stoi:
            raise ValueError("Padding symbol is not in the vocabulary.")
        return self.stoi[SpecialVocabSymbols.PAD]

    def __iadd__(self, values: Union["Vocab", Iterable]):
        """Adds additional values or another Vocab to this Vocab.

        Parameters
        ----------
        values : Iterable or Vocab
            Values to be added to this Vocab.
            If Vocab, all of the token frequencies and specials from that Vocab will be
            added to this Vocab.

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
        if self.finalized:
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
                    " Try adding a non-finalized vocab or or a Vocab with "
                    "`keep_freqs` enabled."
                )

            # unique is used instead of set to somewhat preserve ordering
            self.specials = list(unique(chain(self.specials, other_vocab.specials)))
            self._has_specials = len(self.specials) > 0
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

    def __add__(self, values: Union["Vocab", Iterable]):
        """Method allows a vocabulary to be added to current vocabulary or
        that a set of values is added to the vocabulary.

        If max_size if None for any of the two Vocabs, the max_size of the resulting Vocab
        will also be None. If they are both defined, the max_size of the resulting Vocab
        will be the sum of max_sizes.

        Parameters
        ----------
        values : Iterable or Vocab
            If Vocab, a new Vocab will be created containing all of the special symbols
            and tokens from both Vocabs.
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
            specials = tuple(unique(chain(self.specials, other_vocab.specials)))

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
                self.finalized
                and other_vocab.finalized
                and self._keep_freqs
                and other_vocab._keep_freqs
                or not self.finalized
                and not other_vocab.finalized
            ):
                # If both have _freqs add them together
                new_freqs = self._freqs + other_vocab._freqs
                new_vocab._freqs = new_freqs
                if self.finalized:
                    new_vocab.finalize()
                return new_vocab

            elif self.finalized and other_vocab.finalized:
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
            if self.finalized:
                new_vocab.finalize()
            return new_vocab

    def finalize(self):
        """Method finalizes vocab building. It also releases frequency counter
        if user set not to keep them.

        Raises
        ------
        RuntimeError
            If the vocab is already finalized.
        """
        if self.finalized:
            _LOGGER.warning(
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
            self._freqs = None  # release memory
        self.finalized = True
        _LOGGER.debug("Vocabulary is finalized.")

    def numericalize(self, data):
        """Method numericalizes given tokens.

        Parameters
        ----------
        data : iter(str)
            iterable collection of tokens

        Returns
        -------
        numericalized_vector : array-like
            numpy array of numericalized tokens

        Raises
        ------
        RuntimeError
            If the vocabulary is not finalized.
        """
        if not self.finalized:
            raise RuntimeError(
                "Cannot numericalize if the vocabulary has not been "
                "finalized because itos and stoi are not yet built."
            )
        if self.filter_unk:
            # Filter out words that are not present in the vocabulary
            return np.array([self.stoi[token] for token in data if token in self.stoi])
        else:
            # Keep unknown specials
            return np.array([self.stoi[token] if token in self.stoi else self.unk_index
                             for token in data])

    def reverse_numericalize(self, numericalized_data: Iterable):
        """Transforms an iterable containing numericalized data into a list of tokens.
        The tokens are read from this Vocab's itos and no additional processing is done.

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
        if not self.finalized:
            raise RuntimeError(
                "Cannot reverse numericalize if the vocabulary has not been "
                "finalized because itos and stoi are not yet built."
            )
        return [self.itos[i] for i in numericalized_data]

    @property
    def has_specials(self):
        """Property that checks if the vocabulary contains special symbols.

        Returns
        -------
        flag : bool
            true if the vocabulary has special symbols, false otherwise.
        """
        return self._has_specials

    def __len__(self):
        """Method calculates vocab lengths including special symbols.

        Returns
        -------
        length : int
            vocab size including special symbols
        """
        if self.finalized:
            return len(self.itos)
        return len(self._freqs)

    def __eq__(self, other):
        """Two vocabs are same if they have same finalization status, their
        stoi and itos mappings are same and their frequency counters are same.

        Parameters
        ----------
        other : object
            object for which we want to knwo equality propertiy

        Returns
        -------
        equal : bool
            true if two vocabs are same, false otherwise
        """
        if not isinstance(other, Vocab):
            return False
        if self.finalized != other.finalized:
            return False
        if self._freqs != other._freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __iter__(self):
        """Method returns iterator over vocabulary, if the vocabulary is not
        finalized iteration is done over frequency counter and special symbols
        are not included, otherwise it is performed on itos and special
        symbols are included.

        Returns
        -------
        iter
            iterator over vocab tokens
        """
        if not self.finalized:
            return iter(self._freqs.keys())
        return iter(self.itos)

    def __repr__(self):
        return f"{type(self).__name__}[finalized: {self.finalized}, size: {len(self)}]"

    def __getitem__(self, token):
        """Returns the token index of the passed token. If the passed token has no index,
        UNK token index is returned.
        Otherwise, an exception is raised.

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

        if not self.finalized:
            raise RuntimeError(
                "Cannot numericalize if the vocabulary has not been "
                "finalized because itos and stoi are not yet built."
            )

        return self.stoi[token]
