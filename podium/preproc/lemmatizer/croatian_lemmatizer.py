"""Module Croatian Lemmatizer loads already prepared
lemmatizer dictionaries. It can return all possible word
inflections for a lemma, or return the lemma of any
word inflexion for the Croatian language."""
import functools
import logging
import os

from podium.preproc.util import (capitalize_target_like_source,
                                 uppercase_target_like_source)
from podium.storage.resources.large_resource import (init_scp_large_resource_from_kwargs,
                                                     SCPLargeResource)

_LOGGER = logging.getLogger(__name__)


class CroatianLemmatizer():
    """Class for lemmatizing words and fetching word
    inflections for a given lemma

    Attributes
    ----------
    BASE_FOLDER : str
        folder to download lemmatizer resources
    MOLEX14_LEMMA2WORD : str
        dictionary file path containing lemma to words mappings
    MOLEX14_WORD2LEMMA : str
        dictionary file path containing word to lemma mappings
    """

    BASE_FOLDER = "molex"
    MOLEX14_LEMMA2WORD = os.path.join(BASE_FOLDER, "molex14_lemma2word.txt")
    MOLEX14_WORD2LEMMA = os.path.join(BASE_FOLDER, "molex14_word2lemma.txt")

    def __init__(self, **kwargs):
        """Creates a lemmatizer object.

        Parameters
        ----------
        **kwargs : dict
            Additional key-value parameters.
            Forwards kwargs to SCPLargeResource
        """
        self.__word2lemma_dict = None
        self.__lemma2word_dict = None

        CroatianLemmatizer.MOLEX14_LEMMA2WORD = os.path.join(
            SCPLargeResource.BASE_RESOURCE_DIR, self.MOLEX14_LEMMA2WORD)
        CroatianLemmatizer.MOLEX14_WORD2LEMMA = os.path.join(
            SCPLargeResource.BASE_RESOURCE_DIR, self.MOLEX14_WORD2LEMMA)

        # automatically downloads molex resources
        # defaults should work for linux and access to djurdja.fer.hr
        init_scp_large_resource_from_kwargs(
            resource=self.BASE_FOLDER, uri="/storage/molex/molex.zip", user_dict=kwargs,
            archive="zip", scp_host="djurdja.takelab.fer.hr"
        )

    @capitalize_target_like_source
    def lemmatize_word(self, word, **kwargs):
        """Returns the lemma for the provided word if
        there is a word in a dictionary. If not found,
        returns the word if none_if_oov=False, and None otherwise.

        Parameters
        ----------
        word : str
            A Croatian language word to lemmatize
        none_if_oov : bool
            A flag indicating whether to return None or the original
            word if the word is not in a dictionary. Passed via kwargs.

        Returns
        -------
        str
            Lemma of the word uppercased at the same chars as
            the original word
        """
        try:
            none_if_oov = kwargs['none_if_oov']
        except KeyError:
            none_if_oov = False

        try:
            return self._word2lemma[word]
        except KeyError:
            _LOGGER.info("Word is being returned instead of lemma.")
            return word if not none_if_oov else None

    def get_words_for_lemma(self, lemma):
        """Returns a list of words that shares the provided lemma.

        Parameters
        ----------
        word : str
            Word lemma to find words that share this lemma

        Returns
        -------
        list(str)
            List of words that share the lemma provided
            uppercased at same chars as lemma provided

        Raises
        ------
        ValueError
            If no words for the provided lemma are found
        """
        try:
            words = self._lemma2word[lemma.lower()]
            is_lower = lemma.islower()
            return [
                w
                if is_lower
                else uppercase_target_like_source(lemma, w)
                for w in words
            ]
        except KeyError:
            error_msg = "No words found for lemma {}".format(lemma)
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

    @property
    def _word2lemma(self):
        """Lazy loading of word2lemma dict"""
        if not self.__word2lemma_dict:
            self.__word2lemma_dict = self._get_word2lemma_dict()
        return self.__word2lemma_dict

    @property
    def _lemma2word(self):
        """Lazy loading of lemma2word dict"""
        if not self.__lemma2word_dict:
            self.__lemma2word_dict = self._get_lemma2word_dict()
        return self.__lemma2word_dict

    def _get_word2lemma_dict(self):
        molex_dict = {}
        with open(self.MOLEX14_WORD2LEMMA, encoding='utf-8') as fp_word:
            for line in fp_word.readlines():
                word, lemma = line.split(" ")
                molex_dict[word] = lemma.rstrip()
        return molex_dict

    def _get_lemma2word_dict(self):
        molex_dict = {}
        with open(self.MOLEX14_LEMMA2WORD, encoding='utf-8') as fp_lemma:
            for line in fp_lemma.readlines():
                sline = line.split('#')
                lemma, words = sline
                words = words.rstrip().split(',')
                molex_dict[lemma] = words
        return molex_dict


def _lemmatizer_posttokenized_hook(
        raw, tokenized, lemmatizer):
    """Lemmatizer postokenized hook that can be used in field processing.
    It is intented for the user to use `get_croatian_lemmatizer_hook`
    instead of this function as it hides Lemmatizer initialization and ensures
    that the constructor is called once.

    Parameters
    ----------
    raw : str
        raw field content
    tokenized : iter(str)
        iterable of tokens that needs to be lemmatized
    lemmatizer : CroatianLematizer
        croatian lemmatizer instance

    Returns
    -------
    raw, tokenized : tuple(str, iter(str))
        Method returns unchanged raw and stemmed tokens.
    """
    return raw, [lemmatizer.lemmatize_word(token) for token in tokenized]


def get_croatian_lemmatizer_hook(**kwargs):
    """Method obtains croatian lemmatizer hook.

    Parameters
    ----------
    kwargs : dict
        Croatian lemmatizer arguments.
    """
    return functools.partial(_lemmatizer_posttokenized_hook,
                             lemmatizer=CroatianLemmatizer(**kwargs))
