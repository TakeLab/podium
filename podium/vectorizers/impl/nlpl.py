import os

from podium.storage import LargeResource
from podium.vectorizers.vectorizer import WordVectors, zeros_default_vector


class NlplVectorizer(WordVectors):
    NAME = "nlpl_vectors"
    URL = "http://vectors.nlpl.eu/repository/11/36.zip"
    ARCHIVE_TYPE = "zip"
    VECTOR_FILE_NAME = "model.txt"

    def __init__(
        self,
        default_vector_function=zeros_default_vector,
        cache_path=None,
        max_vectors=None,
    ):
        LargeResource(
            **{
                LargeResource.RESOURCE_NAME: NlplVectorizer.NAME,
                LargeResource.URI: NlplVectorizer.URL,
                LargeResource.ARCHIVE: NlplVectorizer.ARCHIVE_TYPE,
            }
        )

        vector_filepath = os.path.join(
            LargeResource.BASE_RESOURCE_DIR,
            NlplVectorizer.NAME,
            NlplVectorizer.VECTOR_FILE_NAME,
        )

        super().__init__(
            path=vector_filepath,
            default_vector_function=default_vector_function,
            cache_path=cache_path,
            max_vectors=max_vectors,
            binary=False,
            encoding="ISO-8859-1",
        )
