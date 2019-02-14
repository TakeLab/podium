"""Module vectorizer offers classes for vectorizing tokens.
 Interface of implemented concrete vectorizers is given in Vectorizer class.

"""
import os
import six

import numpy as np

from abc import ABC, abstractmethod


def zeros_default_vector(token, dim):
    """Function for creating default vector for given token in form of zeros
    array. Dimension of returned array is equal to given dim.

    Parameters
    ----------
    token : str
        string token from vocabulary
    dim : int
        vector dimension

    Returns
    -------
    vector : array-like
        zeros vector with given dimension
    """
    if dim is None:
        raise ValueError("Dim mustn't be None,"
                         " given token={}, dim={}".format(token, dim))
    return np.zeros(dim)


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
        self._path = path
        self._default_vector_function = default_vector_function
        self._cache_path = cache_path
        self._max_vectors = max_vectors

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
        vocab : iterable object
            vocabulary with unique words

        Raises
        ------
        IOError
            if there was a problem while reading vectors from instance path
        ValueError
            if given path is not a valid path or given vocab is none
            or if the vector values in vector storage cannot be casted to float
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
        RuntimeError
            if vector storage is not initialized
        """
        pass

    def __getitem__(self, key):
        return self.token_to_vector(key)

    @abstractmethod
    def get_vector_dim(self):
        """"Method returns vector dimension.

        Returns
        -------
        dim : int
            vector dimension

        Raises
        ------
        RuntimeError
            if vector storage is not initialized
        """
        pass

    def get_embedding_matrix(self, itos):
        """Method constructs embedding matrix.

        Parameters
        ----------
        itos : iter(str)
            collection of tokens for creation of embedding matrix
            default use case is to give this function itos list

        Raises
        ------
        RuntimeError
            if vector storage is not initialized
        """
        return np.vstack([self.token_to_vector(token) for token in itos])


class BasicVectorStorage(VectorStorage):
    """ Basic implementation of VectorStorage that handles loading vectors from
    system storage.

    Attributes
    ----------
    _vectors : dict
        dictionary offering word to vector mapping
    _dim : int
        vector dimension
    _initialized : bool
        has the vector storage been initialized by loading vectors
    _binary : bool
        if True, the file is read as a binary file.
        Else, it's read as a plain utf-8 text file.

    """

    def __init__(self, path, default_vector_function=zeros_default_vector,
                 cache_path=None, max_vectors=None, binary=True):
        self._vectors = dict()
        self._dim = None
        self._initialized = False
        self._binary = binary
        super().__init__(
            path=path,
            default_vector_function=default_vector_function,
            cache_path=cache_path,
            max_vectors=max_vectors)

    def __len__(self):
        return len(self._vectors)

    def load_all(self):
        self._load_vectors()

    def load_vocab(self, vocab):
        if vocab is None:
            raise ValueError("Vocab mustn't be None")
        self._load_vectors(vocab=vocab)

    def token_to_vector(self, token):
        if not self._initialized:
            raise RuntimeError("VectorStorage is not initialized."
                               "Use load_all or load_vocab function"
                               " to initialize.")
        if token is None:
            raise ValueError("Token mustn't be None")
        if token not in self._vectors \
           and self._default_vector_function is not None:
            return self._default_vector_function(token, self._dim)
        return self._vectors[token]

    def _cache_vectors(self):
        """Method for caching loaded vectors to cache_dir."""
        with open(self._cache_path, "wb") as cache_file:
            for word in self._vectors:
                vector_values_string = " ".join(map(str, self._vectors[word]))
                cache_file.write("{} {}\n".format(word, vector_values_string)
                                 .encode('utf-8'))

    @staticmethod
    def _decode_word(word):
        """Method tries to decode binary word as utf-8 raises UnicodeError if
        fails.

        Parameters
        ----------
        word : str
            binary type string that needs to be decoded

        Returns
        -------
        decoded : str
            decoded word

        Raises
        ------
        UnicodeDecodeError
            if given word cannot be decoded in unicode
        """
        if isinstance(word, six.binary_type):
            decoded = word.decode('utf-8')
            return decoded

    def _load_vectors(self, vocab=None):
        """Internal method for loading vectors. It combines vocab vectors
        loading and all vectors loading.

        Parameters
        ----------
        vocab : iterable object
            vocabulary with unique words

        Raises
        ------
        UnicodeDecodeError
            if given word cannot be decoded in unicode
        RuntimeError
            if file contains empty line or if it contains more that
            one header line
        ValueError
            if given path is not a valid path or if the line in vector storage
            cannot be casted to float
        """
        self._check_path()
        curr_path = self._path if self._cache_path is None \
            or not os.path.exists(self._cache_path) else self._cache_path

        if vocab is not None and not isinstance(vocab, set):
            vocab = set(vocab)

        open_mode, split_delimiter = ('rb', b' ') if self._binary else ('r', ' ')

        with open(curr_path, open_mode) as vector_file:

            vectors_loaded = 0
            header_lines = 0
            for line in vector_file:
                stripped_line = line.rstrip()
                if not stripped_line:
                    raise RuntimeError("File contains empty lines")

                word, vector_entries_str = stripped_line.split(split_delimiter, 1)
                vector_entry = np.fromstring(string=vector_entries_str,
                                             dtype=float, sep=' ')
                # will throw ValueError if vector_entries_str cannot be casted

                if self._dim is None and len(vector_entry) > 1:
                    self._dim = len(vector_entry)
                elif len(vector_entry) == 1:
                    header_lines += 1
                    if header_lines > 1:
                        raise RuntimeError("Found more than one header line")
                    continue  # probably a header, reference torch text
                # second reference:
                # https://radimrehurek.com/gensim/scripts/glove2word2vec.html
                elif self._dim != len(vector_entry):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, "
                        "but previously read vectors have {} dimensions. "
                        "All vectors must have the same "
                        "number of dimensions.".format(word,
                                                       len(vector_entry),
                                                       self._dim))

                if self._binary:
                    word = self._decode_word(word)

                if vocab is not None and word not in vocab:
                    continue

                self._vectors[word] = vector_entry
                vectors_loaded += 1
                if vectors_loaded == self._max_vectors:
                    break

            if self._cache_path is not None \
                    and not os.path.exists(self._cache_path):
                self._cache_vectors()
        self._initialized = True

    def get_vector_dim(self):
        if not self._initialized:
            raise RuntimeError("Cannot obtain vector dimension until"
                               " vector storage is initialized.")
        return self._dim

    def _check_path(self):
        """Internal method for determining if instance paths are in supported
        state. It enforces that both paths cannot be None and that not None
        path must exist unless if it is used for caching loaded vectors.
        """
        if self._path is None and self._cache_path is None:
            raise ValueError("Given vectors and cache paths mustn't"
                             " be both None")

        if self._path is not None and not os.path.exists(self._path):
            raise ValueError("Given vectors path doesn't exist.")

        if self._path is None and self._cache_path is not None\
           and not os.path.exists(self._cache_path):
            raise ValueError("Given cache path doesn't exist.")
