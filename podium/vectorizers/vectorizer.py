"""
Module vectorizer offers classes for vectorizing tokens.

Interface of implemented concrete vectorizers is given in Vectorizer class.
"""
import os
from abc import ABC, abstractmethod

import numpy as np

from podium.utils.general_utils import repr_type_and_attrs


def zeros_default_vector(token, dim):
    """
    Function for creating default vector for given token in form of zeros array.
    Dimension of returned array is equal to given dim.

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

    Raises
    ------
        If dim is None.
    """
    if dim is None:
        raise ValueError(
            "Can't create zeros default vector with dimension "
            f"equal to None. Given token= {token}, dim={dim}"
        )
    return np.zeros(dim)


def random_normal_default_vector(token, dim):
    """
    Draw a random vector from a standard normal distribution. Dimension of
    returned array is equal to given dim.

    Parameters
    ----------
    token : str
        string token from vocabulary
    dim : int
        vector dimension

    Returns
    -------
    vector : array-like
        sampled from normal distribution with given dimension
    """
    if dim is None:
        raise ValueError(
            "Can't create random normal vector with dimension "
            f"equal to None. Given token={token}, dim={dim}"
        )
    return np.random.randn(dim)


class VectorStorage(ABC):
    """
    Interface for classes that can vectorize token.

    One example of such vectorizer is word2vec.
    """

    def __init__(
        self, path, default_vector_function=None, cache_path=None, max_vectors=None
    ):
        """
        Vectorizer base class constructor.

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
            maximum number of vectors to load in memory
        """
        self._path = path
        self._default_vector_function = default_vector_function
        self._cache_path = cache_path
        self._max_vectors = max_vectors

    @abstractmethod
    def load_all(self):
        """
        Method loads all vectors stored in instance path to the vectors.

        Raises
        ------
        IOError
            If there was a problem while reading vectors from instance path.
        ValueError
            If instance path is not a valid path.
        RuntimeError
            If different vector size is detected while loading vectors.
        """
        pass

    @abstractmethod
    def load_vocab(self, vocab):
        """
        Method loads vectors for tokens in vocab stored in given path to the
        instance.

        Parameters
        ----------
        vocab : iterable object
            vocabulary with unique words

        Raises
        ------
        IOError
            If there was a problem while reading vectors from instance path.
        ValueError
            If given path is not a valid path or given vocab is none
            or if the vector values in vector storage cannot be casted to float.
        RuntimeError
            If different vector size is detected while loading vectors.
        """
        pass

    @abstractmethod
    def token_to_vector(self, token):
        """
        Method obtains vector for given token.

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
            If given token doesn't have vector representation and default
            vector function is not defined (None).
        ValueError
            If given token is None.
        RuntimeError
            If vector storage is not initialized.
        """
        pass

    def __getitem__(self, key):
        return self.token_to_vector(key)

    @abstractmethod
    def get_vector_dim(self):
        """
        Method returns vector dimension.

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

    @abstractmethod
    def __len__(self):
        """
        Method returns number of vectors in vector storage.

        Returns
        -------
        len : int
            number of loaded vectors in vector storage
        """
        pass

    def get_embedding_matrix(self, vocab=None):
        """
        Method constructs embedding matrix.

        Note: From python 3.6 dictionaries preserve insertion order
        https://docs.python.org/3.6/whatsnew/3.6.html#other-language-changes

        Parameters
        ----------
        vocab : iter(token)
            collection of tokens for creation of embedding matrix
            default use case is to give this function vocab or itos list
            or `None` if you wish to retrieve all loaded vectors. In case
            `None` is passed as argument, the order of vectors is the same
            as the insertion order of loaded vectors in `VectorStorage`.

        Raises
        ------
        RuntimeError
            If vector storage is not initialized.
        """
        if vocab is None:
            # Retrieve all loaded vectors
            vocab = list(self._vectors.keys())
        return np.vstack([self.token_to_vector(token) for token in vocab])

    def __repr__(self):
        attrs = {"size": len(self)}
        return repr_type_and_attrs(self, attrs)


class WordVectors(VectorStorage):
    """
    Basic implementation of VectorStorage that handles loading vectors from
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

    def __init__(
        self,
        path,
        default_vector_function=random_normal_default_vector,
        cache_path=None,
        max_vectors=None,
        encoding="utf-8",
        binary=True,
    ):
        self._vectors = {}
        self._dim = None
        self._initialized = False
        self._binary = binary
        self._encoding = encoding if not binary else None
        super().__init__(
            path=path,
            default_vector_function=default_vector_function,
            cache_path=cache_path,
            max_vectors=max_vectors,
        )

    def __len__(self):
        return len(self._vectors)

    def load_all(self):
        self._load_vectors()

    def load_vocab(self, vocab):
        if vocab is None:
            raise ValueError("Cannot load vectors for vocab because given vocab is None.")
        self._load_vectors(vocab=vocab)
        return self.get_embedding_matrix(vocab)

    def token_to_vector(self, token):
        if not self._initialized:
            raise RuntimeError(
                "Vector storage is not initialized so it cannot"
                " transform token to vector. Use load_all or "
                " load_vocab function to initialize."
            )
        if token is None:
            raise ValueError(
                "User gave None token to be converted to vector"
                ", but None is not a valid token."
            )
        if token not in self._vectors and self._default_vector_function is not None:
            return self._default_vector_function(token, self._dim)
        return self._vectors[token]

    def _cache_vectors(self):
        """
        Method for caching loaded vectors to cache_dir.
        """
        with open(self._cache_path, "wb") as cache_file:
            for word in self._vectors:
                vector_values_string = " ".join(map(str, self._vectors[word]))
                cache_file.write(f"{word} {vector_values_string}\n".encode("utf-8"))

    @staticmethod
    def _decode_word(word):
        """
        Method tries to decode binary word as utf-8 raises UnicodeError if
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
        AssertionError
            If word is not an instance of bytes.
        UnicodeDecodeError
            If given word cannot be decoded in unicode.
        """
        assert isinstance(word, bytes)
        decoded = word.decode("utf-8")
        return decoded

    def _load_vectors(self, vocab=None):
        """
        Internal method for loading vectors. It combines vocab vectors loading
        and all vectors loading.

        Parameters
        ----------
        vocab : iterable object
            vocabulary with unique words

        Raises
        ------
        UnicodeDecodeError
            If given word cannot be decoded in unicode.
        RuntimeError
            If file contains empty line or if it contains more that
            one header line.
        ValueError
            If given path is not a valid path or if the line in vector storage
            cannot be casted to float.
        """
        self._check_path()
        curr_path = (
            self._path
            if self._cache_path is None or not os.path.exists(self._cache_path)
            else self._cache_path
        )

        if vocab is not None and not isinstance(vocab, set):
            vocab = set(vocab)

        open_mode, split_delimiter = ("rb", b" ") if self._binary else ("r", " ")
        with open(curr_path, open_mode, encoding=self._encoding) as vector_file:

            vectors_loaded = 0
            header_lines = 0
            for line in vector_file:
                stripped_line = line.rstrip()
                if not stripped_line:
                    raise RuntimeError(
                        "Vectors file contains empty lines which is not supported."
                    )

                word, vector_entries_str = stripped_line.split(split_delimiter, 1)
                vector_entry = np.fromstring(
                    string=vector_entries_str, dtype=float, sep=" "
                )
                # will throw ValueError if vector_entries_str cannot be casted

                if self._dim is None and len(vector_entry) > 1:
                    self._dim = len(vector_entry)
                elif len(vector_entry) == 1:
                    header_lines += 1
                    if header_lines > 1:
                        raise RuntimeError(
                            "Found more than one header line in vectors file."
                        )
                    continue  # probably a header, reference torch text
                # second reference:
                # https://radimrehurek.com/gensim/scripts/glove2word2vec.html
                elif self._dim != len(vector_entry):
                    raise RuntimeError(
                        f"Vector for token {word} has {len(vector_entry)} "
                        "dimensions, but previously read vectors have "
                        f"{self._dim} dimensions. All "
                        "vectors must have the same number of dimensions."
                    )

                if self._binary:
                    word = self._decode_word(word)

                if vocab is not None and word not in vocab:
                    continue

                self._vectors[word] = vector_entry
                vectors_loaded += 1
                if vectors_loaded == self._max_vectors:
                    break

            if self._cache_path is not None and not os.path.exists(self._cache_path):
                self._cache_vectors()
        self._initialized = True

    def get_vector_dim(self):
        if not self._initialized:
            raise RuntimeError(
                "Vector storage must be initialized to obtain vector dimenstion."
            )
        return self._dim

    def _check_path(self):
        """
        Internal method for determining if instance paths are in supported
        state.

        It enforces that both paths cannot be None and that not None path must
        exist unless if it is used for caching loaded vectors.
        """
        if self._path is None and self._cache_path is None:
            raise ValueError(
                "Error in checking paths that are handed to "
                "load vectors. Given vectors and cache paths "
                "mustn't be both None."
            )

        if self._path is not None and not os.path.exists(self._path):
            raise ValueError(
                "Error in checking paths that are handed to "
                "load vectors. Given vectors path doesn't"
                " exist. If you want to use only cached path "
                "set path to None."
            )

        if (
            self._path is None
            and self._cache_path is not None
            and not os.path.exists(self._cache_path)
        ):
            raise ValueError(
                "Error in checking paths that are handed to "
                "load vectors. Given cache path doesn't exist."
                " User needs to specify valid path or existing "
                "cache path."
            )
