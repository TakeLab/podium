"""Module Croatian Lemmatizer loads already prepared
lemmatizer dictionaries. It can return all possible word
inflections for a lemma, or return the lemma of any
word inflexion for the Croatian language."""

from takepod.storage.large_resource import LargeResource, SCPLargeResource
import getpass
import os


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

    def __init__(self):
        """Creates a lemmatizer object."""
        self.__word2lemma_dict = None
        self.__lemma2word_dict = None

        # automatically downloads molex resources
        # defaults should work for linux and access to djurdja.fer.hr
        # if your username is same as one on djurdja
        SCPLargeResource(**{
            LargeResource.URI: "/storage/molex/molex.zip",
            LargeResource.RESOURCE_NAME: self.BASE_FOLDER,
            LargeResource.ARCHIVE: "zip",
            SCPLargeResource.SCP_HOST_KEY: "djurdja.takelab.fer.hr",
            SCPLargeResource.SCP_USER_KEY: getpass.getuser(),
            SCPLargeResource.SCP_PASS_KEY: None,
            SCPLargeResource.SCP_PRIVATE_KEY: None
        })

    def lemmatize_word(self, word):
        """Returns the lemma for the provided word if
        there is a word in a dictionary, otherwise returns the word.

        Parameters
        ----------
        word : str
            A Croatian language word to lemmatize

        Returns
        -------
        str
            Lemma of the word uppercased at the same chars as
            the original word
        """

        try:
            lemma = self._word2lemma[word.lower()]
            return self._uppercase_target_like_source(word, lemma)
        except KeyError:
            # TODO: insert log statement that a word is being returned
            return word

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
            return [
                self._uppercase_target_like_source(lemma, w) for w in words
            ]
        except KeyError:
            raise ValueError("No words found for lemma {}".format(lemma))

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

    def _uppercase_target_like_source(self, source, target):
        uppercased_target = ''.join([
            target[i].upper()
            if s.isupper() and s.lower() == target[i] else target[i]
            for i, s in zip(range(len(target)), source)
        ])
        uppercased_target += target[len(source):]
        return uppercased_target

    def _get_word2lemma_dict(self):
        molex_dict = {}
        with open(self.MOLEX14_WORD2LEMMA, encoding='utf-8') as f:
            for line in f.readlines():
                word, lemma = line.split(" ")
                molex_dict[word] = lemma.rstrip()
        return molex_dict

    def _get_lemma2word_dict(self):
        molex_dict = {}
        with open(self.MOLEX14_LEMMA2WORD, encoding='utf-8') as f:
            for line in f.readlines():
                sline = line.split('#')
                lemma, words = sline
                words = words.rstrip().split(',')
                molex_dict[lemma] = words
        return molex_dict
