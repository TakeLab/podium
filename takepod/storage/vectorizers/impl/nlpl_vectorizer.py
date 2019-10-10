import os

from takepod.storage.vectorizers.vectorizer import BasicVectorStorage \
    , zeros_default_vector
from takepod.storage import LargeResource


class NlplVectorizer(BasicVectorStorage):
    NAME = 'nlpl_vectors'
    URL = 'http://vectors.nlpl.eu/repository/11/36.zip'
    ARCHIVE_TYPE = 'zip'
    VECTOR_FILE_NAME = 'model.txt'

    def __init__(self,
                 default_vector_function=zeros_default_vector,
                 cache_path=None, max_vectors=None):
        LargeResource(**{
            LargeResource.RESOURCE_NAME: NlplVectorizer.NAME,
            LargeResource.URI: NlplVectorizer.URL,
            LargeResource.ARCHIVE: NlplVectorizer.ARCHIVE_TYPE
        })

        vector_filepath = os.path.join(LargeResource.BASE_RESOURCE_DIR,
                                       NlplVectorizer.NAME,
                                       NlplVectorizer.VECTOR_FILE_NAME)

        super().__init__(path=vector_filepath,
                         default_vector_function=default_vector_function,
                         cache_path=cache_path,
                         max_vectors=max_vectors,
                         binary=False)
