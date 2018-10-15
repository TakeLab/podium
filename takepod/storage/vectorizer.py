"""Module vectorizer offers classes for vectorizing tokens.
 Interface of implemented concrete vectorizers is given in Vectorizer class.

"""

from abc import ABC, abstractmethod


class Vectorizer(ABC):
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

    @abstractmethod
    def vector_to_token(self, vector):
        """Method obtains token for given vector.

        Parameters
        ----------
        vector : array_like
            vector representation of a token

        Returns
        -------
        token : str
            token from vocabulary

        Raises
        ------
        ValueError
            if given vector is None
        KeyError
            if given vector doesn't have token representation

        """
        pass

    def __getitem__(self, key):
        return self.token_to_vector(key)
