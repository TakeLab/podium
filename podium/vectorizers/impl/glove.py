import os

from podium.storage import LargeResource
from podium.vectorizers.vectorizer import WordVectors, random_normal_default_vector


class GloVe(WordVectors):
    """
    Class represents concrete vector storage for GloVe vectors described in
    https://nlp.stanford.edu/projects/glove/ . Class contains a Large resource
    so that vectors could be automatically downloaded on first use.

    Attributes
    ----------
    NAME_URL_MAPPING : dict(str, str)
        dictionary that maps glove instance name to download url
    NAME_DIM_MAPPING : dict(str, set(str))
        dictionary that maps glove instance name to available vector dimensions
        for given instance
    _NAME_FILE_MAPPING : dict(str, str)
        dictionary that maps glove instance name to filenames available in vectors
        folder
    _ARCHIVE_TYPE : str
        type of arhive in which the vectors are stored while downloading
    _BINARY : bool
        defines if the vectors are stored in binary format or not. glove vectors
        are stored in binary format
    """

    NAME_URL_MAPPING = {
        "glove-common-42b": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
        "glove-common-840b": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
        "glove-twitter": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "glove-wikipedia": "http://nlp.stanford.edu/data/glove.6B.zip",
    }
    NAME_DIM_MAPPING = {
        "glove-common-42b": {
            300,
        },
        "glove-common-840b": {
            300,
        },
        "glove-twitter": {25, 50, 100, 200},
        "glove-wikipedia": {50, 100, 200, 300},
    }
    _NAME_FILE_MAPPING = {
        "glove-common-42b": "glove.42B",
        "glove-common-840b": "glove.840B.",
        "glove-twitter": "glove.twitter.27B",
        "glove-wikipedia": "glove.6B",
    }

    _ARCHIVE_TYPE = "zip"
    _BINARY = True

    def __init__(
        self,
        name="glove-wikipedia",
        dim=300,
        default_vector_function=random_normal_default_vector,
        cache_path=None,
        max_vectors=None,
    ):
        """
        GloVe constructor that initializes vector storage and downloads vectors
        if necessary.

        Parameters
        ----------
        name : str
            name of glove vectors instance, available names are available in
            NAME_URL_MAPPING dictionary
        dim : int
            vectors dimension, available dimensions are listed in NAME_DIM_MAPPING
            dictionary
        default_vector_function : callable, optional
            which vector should be returned if vectorizer doesn't have
            representation for given token. If None and token doesn't
            exists an error is raised while obtaining a vector
        cache_path : str
            path for caching vectors, useful if not loading all vectors from file
            by either loading some arbitrary number of vectors (see max_vectors) or
            by loading vectors for vocabulary.
        max_vectors : int
            maximum number of vectors to load in memory

        Raises
        ------
        ValueError
            If given name is not in NAME_URL_MAPPING keys or if the given vectors
            dimension is not available. Supported dimensions are available in
            NAME_DIM_MAPPING dictionary.
        """
        if name not in GloVe.NAME_URL_MAPPING.keys():
            raise ValueError(
                "Given name not supported, supported names are "
                f"{GloVe.NAME_URL_MAPPING.keys()}"
            )
        if dim not in GloVe.NAME_DIM_MAPPING[name]:
            raise ValueError(
                "Unsupported dimension for given glove instance, "
                f"{name} GloVe instance has following supported dimensions "
                f"{GloVe.NAME_DIM_MAPPING[name]}"
                ""
            )

        url = GloVe.NAME_URL_MAPPING[name]
        LargeResource(
            **{
                LargeResource.RESOURCE_NAME: name,
                LargeResource.ARCHIVE: GloVe._ARCHIVE_TYPE,
                LargeResource.URI: url,
            }
        )

        file_name = f"{GloVe._NAME_FILE_MAPPING[name]}.{dim}d.txt"
        path = os.path.join(LargeResource.BASE_RESOURCE_DIR, name, file_name)

        vectors_kwargs = {
            "default_vector_function": default_vector_function,
            "cache_path": cache_path,
            "max_vectors": max_vectors,
            "path": path,
            "binary": GloVe._BINARY,
        }
        super(GloVe, self).__init__(**vectors_kwargs)
