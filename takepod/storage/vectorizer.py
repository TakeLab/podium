"""Module vectorizer offers classes for vectorizing tokens.
 Interface of implemented concrete vectorizers is given in Vectorizer class.

"""
import os
from abc import ABC, abstractmethod
import six
from tqdm import tqdm
import numpy as np


class VectorStorage(ABC):
    """Interface for classes that can vectorize token. One example of such
    vectorizer is word2vec.

    """

    def __init__(self, path, default_vector_function=None,
                 cache_path=None, max_vectors=None):
        """ Vectorizer base class constructor.

        Parameters
            ----------
            path : str
                path to stored vectors
            default_vector_function : callable, optional
                which vector should be returned if vectorizer doesn't have
                representation for given token. If None and token doesn't
                exists an error is raised while obtaining a vector
            cache_path : str, optional
                path to cached vectors. Caching vectors should be used when
                using vocab for loading vectors or when limiting number of
                vectors to load
            max_vectors : int, optional
                maximal number of vectors to load in the memory

        """
        self.path = path
        self.default_vector_function = default_vector_function
        self.cache_path = cache_path
        self.max_vectors = max_vectors

    @abstractmethod
    def load_all(self):
        """Method loads all vectors stored in instance path to the vectors.

        Raises
        ------
        IOError
            if there was a problem while reading vectors from instance path
        ValueError
            if instance path is not a valid path
        RuntimeError
            if different vector size is detected while loading vectors

        """
        pass

    @abstractmethod
    def load_vocab(self, vocab):
        """Method loads vectors for tokens in vocab
        stored in given path to the instance.

        Parameters
        ----------
        path : str
            path to stored vectors

        Raises
        ------
        IOError
            if there was a problem while reading vectors from given path
        ValueError
            if given path is not a valid path or given vocab is none
        RuntimeError
            if different vector size is detected while loading vectors

        """
        pass

    @abstractmethod
    def token_to_vector(self, token):
        """Method obtains vector for given token.

        Parameters
        ----------
        token : str
            token from vocabulary

        Returns
        -------
        vector : array_like
            vector representation of given token

        Raises
        ------
        KeyError
            if given token doesn't have vector representation and default
            vector function is not defined (None)
        ValueError
            if given token is None

        """
        pass

    def __getitem__(self, key):
        return self.token_to_vector(key)


class BasicVectorStorage(VectorStorage):
    """ Basic implementation of VectorStorage that handles loading vectors from
    system storage.

    """

    def __init__(self, path, default_vector_function=None,
                 cache_path=None, max_vectors=None):
        super(BasicVectorStorage, self).__init__(
            path=path,
            default_vector_function=default_vector_function,
            cache_path=cache_path,
            max_vectors=max_vectors)
        self.vectors = dict()
        self.dim = None

    def load_all(self):
        self._load_vectors()

    def load_vocab(self, vocab):
        if vocab is None:
            raise ValueError("Vocab mustn't be None")
        self._load_vectors(vocab=vocab)

    def token_to_vector(self, token):
        if token is None:
            raise ValueError("Token mustn't be None")
        return self.vectors[token]

    def _cache_vectors(self):
        pass # TODO implement function

    @staticmethod
    def _decode_word(word):
        try:
            if isinstance(word, six.binary_type):
                decoded = word.decode('utf-8')
                return decoded
        except UnicodeDecodeError:
            pass
        return None

    def _load_vectors(self, vocab=None):
        self._check_path()
        curr_path = self.path if self.path is not None else self.cache_path

        with open(curr_path, 'rb') as vector_file:
            num_lines = 10  # TODO calculate number of lines
            if not self.max_vectors or self.max_vectors > num_lines:
                self.max_vectors = num_lines

            vectors_loaded = 0
            for line in tqdm(vector_file, total=num_lines):
                line_entries = line.rstrip().split(b" ")
                word, vector_entry = line_entries[0], line_entries[1:]

                if self.dim is None and len(vector_entry) > 1:
                    self.dim = len(vector_entry)
                elif len(vector_entry) == 1:
                    continue  # probably a header, reference torch text
                elif self.dim != len(vector_entry):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, "
                        "but previously read vectors have {} dimensions. "
                        "All vectors must have the same "
                        "number of dimensions.".format(word,
                                                       len(vector_entry),
                                                       self.dim))

                word = self._decode_word(word)
                if word is None:
                    continue
                if vocab is not None and word not in vocab:
                    continue
                self.vectors[word] = np.array([float(i) for i in vector_entry])
                vectors_loaded += 1
                if vectors_loaded == self.max_vectors:
                    break
            if self.path is not None and self.cache_path is not None\
               and not os.path.exists(self.cache_path):
                self._cache_vectors()

    def _check_path(self):
        if self.path is None and self.cache_path is None:
            raise ValueError("Given vectors and cache paths mustn't"
                             " be both None")

        if self.path is not None and not os.path.exists(self.path):
            raise ValueError("Given vectors path doesn't exist.")

        if self.cache_path is not None and not os.path.exists(self.path):
            raise ValueError("Given cache path doesn't exist.")
