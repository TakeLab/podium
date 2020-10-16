"""Dataloader for Cornell Movie-Dialogs Corpus, available at
http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html"""
import logging
import os
import re
from collections import namedtuple

from podium.storage import LargeResource


_LOGGER = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    _LOGGER.debug(
        "Problem occured while trying to import pandas. If the library is not "
        "installed visit https://pandas.pydata.org/ for more details."
    )


CornellMovieDialogsNamedTuple = namedtuple(
    typename="CornellMovieDialogsNamedTuple",
    field_names=["titles", "conversations", "lines", "characters", "url"],
)


class CornellMovieDialogsLoader:
    """Class for downloading and parsing the Cornell Movie-Dialogs dataset.

    This class is used for downloading the dataset (if it's not already downloaded) and
    parsing the files in the dataset. If it's not already present
    LargeResource.BASE_RESOURCE_DIR, the dataset is automatically downloaded when an
    instance of the loader is created. The downloaded resources can be parsed using
    the load_dataset method.
    """

    URL = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    ARCHIVE_TYPE = "zip"
    NAME = "cornell_movie_dialogs_corpus"
    DATA_FOLDER_NAME = "cornell movie-dialogs corpus"
    DELIMITER = " +++$+++ "
    ENCODING = "iso-8859-1"

    TITLE_FIELDS = ["movieID", "title", "year", "rating", "votes", "genres"]
    TITLE_FILENAME = "movie_titles_metadata.txt"

    CHARACTERS_FIELDS = [
        "characterID",
        "character",
        "movieID",
        "title",
        "gender",
        "position",
    ]
    CHARACTERS_FILENAME = "movie_characters_metadata.txt"

    LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    LINES_FILENAME = "movie_lines.txt"

    CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
    CONVERSATIONS_FILENAME = "movie_conversations.txt"

    URL_FIELDS = ["movieID", "title", "url"]
    URL_FILENAME = "raw_script_urls.txt"

    def __init__(self):
        """The constructor will check if the dataset is already been downloaded in the
        LargeResource.BASE_RESOURCE_DIR. If the dataset is not present, it will atempt to
        download it.
        """
        LargeResource(
            **{
                LargeResource.RESOURCE_NAME: CornellMovieDialogsLoader.NAME,
                LargeResource.ARCHIVE: CornellMovieDialogsLoader.ARCHIVE_TYPE,
                LargeResource.URI: CornellMovieDialogsLoader.URL,
            }
        )

    def load_dataset(self):
        """Loads and parses all the necessary files from the dataset folder.

        Returns
        -------
        data : CornellMovieDialogsNamedTuple
            tuple that contains dictionaries for 5 types of Cornell movie dialogs data:
            titles, conversations, lines, characters and script urls.
            Fields for every type are defined in class constants.
        """
        titles = self.load_titles()
        conversations = self.load_conversations()
        lines = self.load_lines()
        characters = self.load_characters()
        url = self.load_urls()

        return CornellMovieDialogsNamedTuple(
            titles=titles,
            conversations=conversations,
            lines=lines,
            characters=characters,
            url=url,
        )

    @staticmethod
    def _load_file(file_name, fields, columns_hooks=None):
        """Method loads file from Cornell movie dialogs dataset defined with file name and
        fields that are used in the file.

        Parameters
        ----------
        file_name : str
            string containing file path
        fields : list(str)
            list containing field names
        columns_hooks : dict(str, callable)
            functions that will be called on columns
            variable represents dictionary that maps column name to a function
        """
        data_frame = pd.read_csv(
            filepath_or_buffer=os.path.join(
                LargeResource.BASE_RESOURCE_DIR,
                CornellMovieDialogsLoader.NAME,
                CornellMovieDialogsLoader.DATA_FOLDER_NAME,
                file_name,
            ),
            sep=re.escape(CornellMovieDialogsLoader.DELIMITER),
            encoding=CornellMovieDialogsLoader.ENCODING,
            header=None,
            names=fields,
            engine="python",
        )
        if columns_hooks is not None:
            for column_name in columns_hooks:
                data_frame[column_name] = data_frame[column_name].apply(
                    columns_hooks[column_name]
                )
        return data_frame.to_dict(orient="list")

    def load_titles(self):
        """Method loads file containing movie titles."""
        column_hooks = {}
        column_hooks["genres"] = lambda s: s.strip("[]''").split("', '")
        return self._load_file(
            file_name=CornellMovieDialogsLoader.TITLE_FILENAME,
            fields=CornellMovieDialogsLoader.TITLE_FIELDS,
            columns_hooks=column_hooks,
        )

    def load_conversations(self):
        """Method loads file containing movie conversations."""
        column_hooks = {}
        column_hooks["utteranceIDs"] = lambda s: s.strip("[]''").split("', '")
        return self._load_file(
            file_name=CornellMovieDialogsLoader.CONVERSATIONS_FILENAME,
            fields=CornellMovieDialogsLoader.CONVERSATIONS_FIELDS,
            columns_hooks=column_hooks,
        )

    def load_lines(self):
        """Method loads file containing movie lines."""
        return self._load_file(
            file_name=CornellMovieDialogsLoader.LINES_FILENAME,
            fields=CornellMovieDialogsLoader.LINES_FIELDS,
        )

    def load_characters(self):
        """Method loads file containing movie characters."""
        return self._load_file(
            file_name=CornellMovieDialogsLoader.CHARACTERS_FILENAME,
            fields=CornellMovieDialogsLoader.CHARACTERS_FIELDS,
        )

    def load_urls(self):
        """Method loads file containing movie script urls."""
        return self._load_file(
            file_name=CornellMovieDialogsLoader.URL_FILENAME,
            fields=CornellMovieDialogsLoader.URL_FIELDS,
        )
