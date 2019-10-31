"""Module contains classes related to the vocabulary."""
import logging
from itertools import chain
from collections import Counter
import numpy as np

_LOGGER = logging.getLogger(__name__)


def unique(values):
    seen = set()
    for element in values:
        if element in seen:
            continue
        seen.add(element)
        yield element


class VocabDict(dict):
    """Vocab dictionary class that is used like default dict but without adding missing
    key to the dictionary."""

    def __init__(self, default_factory=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_factory = default_factory

    def __missing__(self, key):
        if self._default_factory is None:
            error_msg = "Default factory is not defined and key is not in " \
                        "the dictionary."
            _LOGGER.error(error_msg)
            raise KeyError
        return self._default_factory()


class SpecialVocabSymbols:
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
    has_specials:
        whether the dictionary contains special symbols
    """

    def __init__(self, max_size=None, min_freq=1,
                 specials=(SpecialVocabSymbols.UNK, SpecialVocabSymbols.PAD),
                 keep_freqs=False):
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
        self.stoi = VocabDict(self._default_unk)
        self.stoi.update({k: v for v, k in enumerate(self.itos)})

        self._max_size = max_size
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

    def _default_unk(self):
        """Method obtains default unknown symbol index. Used for stoi.

        Returns
        -------
        index: int
            index of default unknown symbol

        Raises
        ------
        ValueError
            if unknown symbol is not present in the vocab
        """
        if self._default_unk_index is None:
            error_msg = "Unknown symbol is not present in the vocab but " \
                        "the user asked for the word that isn't in the vocab."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)
        return self._default_unk_index

    def get_freqs(self):
        """Method obtains vocabulary frequencies.

        Returns
        -------
        freq : Counter
            mapping frequency for every word

        Raises
        ------
        RuntimeError
            if the user stated that he doesn't want to keep frequencies
            and the vocab is finalized
        """
        if self.finalized and not self._keep_freqs:
            error_msg = "User specified that frequencies aren't kept in " \
                        "vocabulary but the get_freqs method is called."
            _LOGGER.error(error_msg)
            raise RuntimeError(error_msg)
        return self._freqs

    def pad_symbol(self):
        """Method returns padding symbol index.

        Returns
        -------
        pad_symbol_index : int
            padding symbol index in the vocabullary

        Raises
        ------
        ValueError
            if the padding symbol is not pressent in the vocabulary
        """
        if SpecialVocabSymbols.PAD not in self.stoi:
            error_msg = "Padding symbol is not in the vocabulary so" \
                        " pad_symbol function raises exception."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)
        return self.stoi[SpecialVocabSymbols.PAD]

    def __iadd__(self, values):
        """Method allows a vocabulary to be added to current vocabulary or
        that a set of values is added to the vocabulary.

        Parameters
        ----------
        values : iter or Vocab
            values to be added to the vocabulary

        Returns
        -------
        vocab : Vocab
            returns current Vocab instance to enable chaining

        Raises
        ------
        RuntimeError
            if the current vocab is finalized
        """
        if self.finalized:
            error_msg = "Once finalized, vocabulary cannot be changed."
            _LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        if isinstance(values, str):
            error_msg = "Vocabulary doesn't support adding a string. " \
                        "If you need single word added to vocab," \
                        " you should wrap it to an iterable."
            _LOGGER.error(error_msg)
            raise TypeError(error_msg)
            # if it is a string characters of a string will be added to counter
            # instead of whole string

        if isinstance(values, Vocab):
            self.specials = list(unique(chain(self.specials + values.specials)))
            self._has_specials = len(self.specials) > 0
            self._itos = list(self.specials)

            if values._freqs is None:
                self += values.itos
            else:
                self._freqs += values._freqs  # add freqs to this instance
        else:
            try:
                self._freqs.update(value for value in values
                                   if value not in self.specials)
            except TypeError:
                error_msg = "TypeError exception occurred while adding values " \
                            "to counter object. Vocab supports only adding " \
                            "vocab or iterable to vocab"
                _LOGGER.exception(error_msg)
                raise TypeError(error_msg)
        return self

    def __add__(self,
                values):
        """Method allows a vocabulary to be added to current vocabulary or
        that a set of values is added to the vocabulary.

        Parameters
        ----------
        values : iter or Vocab
            values to be added to the vocabulary

        Returns
        -------
        vocab : Vocab
            returns current Vocab instance to enable chaining

        Raises
        ------
        RuntimeError
            if the current vocab is finalized
        """
        if isinstance(values, Vocab):
            specials = tuple(unique(chain(self.specials, values.specials)))
            if self._max_size is None or values._max_size is None:
                max_size = None
            else:
                max_size = self._max_size + values._max_size
            new_vocab = Vocab(specials=specials,
                              max_size=max_size,
                              min_freq=min(self._min_freq, values._min_freq),
                              keep_freqs=self._keep_freqs or values._keep_freqs)

            if self.finalized and values.finalized and self._keep_freqs \
                    and values._keep_freqs or not self.finalized and not values.finalized:
                new_freqs = self._freqs + values._freqs
                new_vocab._freqs = new_freqs
                if self.finalized:
                    new_vocab.finalize()
                return new_vocab

            elif self.finalized and values.finalized:
                new_vocab += self.itos
                new_vocab += values.itos

                new_vocab.finalize()
                return new_vocab

            else:
                error_msg = "Vocab addition error. When adding up two Vocabs " \
                            "both must be either finalized or not finalized."
                _LOGGER.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            new_vocab = Vocab(specials=self.specials,
                              max_size=self._max_size,
                              min_freq=self._min_freq,
                              keep_freqs=self._keep_freqs)
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
            if the vocab is already finalized
        """
        if self.finalized:
            _LOGGER.warning("Vocabulary is finalized already. "
                            "This should be used only if multiple fields "
                            "use same vocabulary.")
            return

        # construct stoi and itos, sort by frequency
        words_and_freqs = sorted(self._freqs.items(), key=lambda tup: tup[1],
                                 reverse=True)
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
            if the vocabulary is not finalized
        """
        if not self.finalized:
            error_msg = "Cannot numericalize if the vocabulary has not been " \
                        "finalized because itos and stoi are not yet built."
            _LOGGER.error(error_msg)
            raise RuntimeError(error_msg)
        return np.array([self.stoi[token] for token in data])

    @property
    def has_specials(self):
        """
        Method checks if the vocabulary contains special symbols.

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

        Parameter
        ---------
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

    def __str__(self):
        return "{}[finalized: {}, size: {}]".format(
            self.__class__.__name__, self.finalized, len(self))
