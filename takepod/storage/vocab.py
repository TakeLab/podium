"""Module contains classes related to the vocabulary."""
import logging
from collections import Counter, defaultdict
import numpy as np


class SpecialVocabSymbols():
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
                 keep_freqs=False, stop_words=None):
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
        stop_words : set(str), optional
            set of stop words
        """
        self._freqs = Counter()
        self._keep_freqs = keep_freqs
        self._min_freq = min_freq

        specials = () if specials is None else specials
        self._has_specials = len(specials) > 0

        self.itos = list(specials)
        self._default_unk_index = self._init_default_unk_index(specials)
        self.stoi = defaultdict(self._default_unk)
        self.stoi.update({k: v for v, k in enumerate(self.itos)})

        self._max_size = max_size
        self._stop_words = stop_words
        self.finalized = False   # flag to know if we're ready to numericalize

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
            raise ValueError("Unknown symbol is not present in the vocab.")
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
            logging.error("User specified that frequencies shoudn't be "
                          "kept in vocabulary but the get_freqs method "
                          "is called.")
            raise RuntimeError("User specified that the frequencies "
                               "are not kept")
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
            logging.error("Padding symbol is not in the vocabulary so"
                          " pad_symbol function raises exception.")
            raise ValueError("Padding symbol is not in the vocabulary")
        return self.stoi[SpecialVocabSymbols.PAD]

    def __add__(self, values):
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
            logging.warning("Vocabulary is finalized already. "
                            "This should be used only if multiple fields use"
                            " same vocabulary.")
            return self

        if isinstance(values, str):
            raise TypeError("Values mustn't be a string.")
            # if it is a string characters of a string will be added to counter
            # instead of whole string

        if isinstance(values, Vocab):
            self._freqs += values._freqs  # add freqs to this instance
        else:
            try:
                self._freqs.update(values)
            except TypeError:
                
                raise TypeError("Vocab supports only adding vocab or iterable"
                                " to vocab")
        return self

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
        return self.__add__(values)

    def finalize(self):
        """Method finalizes vocab building. It also releases frequency counter
        if user set not to keep them.

        Raises
        ------
        RuntimeError
            if the vocab is already finalized
        """
        if self.finalized:
            raise RuntimeError("Vocab is already finalized.")

        # construct stoi and itos, sort by frequency
        words_and_freqs = sorted(self._freqs.items(), key=lambda tup: tup[1],
                                 reverse=True)
        if self._max_size is None:
            self._max_size = len(words_and_freqs) + len(self.itos)
            # vocab + specials
        for word, freq in words_and_freqs:
            if freq < self._min_freq or len(self.itos) >= self._max_size:
                break
            if self._stop_words and word in self._stop_words:
                continue
            self.itos.append(word)
            self.stoi[word] = len(self.stoi)

        if not self._keep_freqs:
            self._freqs = None  # release memory

        self.finalized = True

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
            raise RuntimeError('Cannot numericalize if the vocabulary has not '
                               'been finalized call `.finalize()`'
                               ' on the Field')
        return np.array([self.stoi[token] for token in data])

    @property
    def has_specials(self):
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
