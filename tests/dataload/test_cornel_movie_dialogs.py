import os
from unittest.mock import patch
import pytest

from podium.storage import LargeResource
from podium.dataload.cornell_movie_dialogs import CornellMovieDialogsLoader

CHARACTERS_DATA = "u0 +++$+++ BIANCA +++$+++ m0 +++$+++ 10 things i hate about you "\
    "+++$+++ f +++$+++ 4\n"\
    "u1 +++$+++ BRUCE +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ ? +++$+++ ?"\
    "\n"\
    "u2 +++$+++ CAMERON +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ m +++$+++"\
    " 3\n"\
    "u3 +++$+++ CHASTITY +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ ? +++$+++"\
    " ?"


EXPECTED_CHARACTERS_DATA = {
    "characterID": ["u0", "u1", "u2", "u3"],
    "character": ["BIANCA", "BRUCE", "CAMERON", "CHASTITY"],
    "movieID": ["m0", "m0", "m0", "m0"],
    "title": ["10 things i hate about you", "10 things i hate about you",
              "10 things i hate about you", "10 things i hate about you"],
    "gender": ["f", "?", "m", "?"],
    "position": ["4", "?", "3", "?"]
}


CONVERSATIONS_DATA = "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']"\
    "\n"\
    "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L198', 'L199']\n"\
    "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L200', 'L201', 'L202', 'L203']\n"\
    "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L204', 'L205', 'L206']\n"\
    "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L207', 'L208']"

EXPECTED_CONVERSATIONS_DATA = {
    "character1ID": ["u0", "u0", "u0", "u0", "u0"],
    "character2ID": ["u2", "u2", "u2", "u2", "u2"],
    "movieID": ["m0", "m0", "m0", "m0", "m0"],
    "utteranceIDs": [['L194', 'L195', 'L196', 'L197'], ['L198', 'L199'],
                     ['L200', 'L201', 'L202', 'L203'], ['L204', 'L205', 'L206'],
                     ['L207', 'L208']]
}


LINES_DATA = "L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!\n"\
    "L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!\n"\
    "L985 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I hope so.\n"\
    "L984 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ She okay?\n"\
    "L925 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Let's go.\n"\
    "L924 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Wow"

EXPECTED_LINES_DATA = {
    "lineID": ["L1045", "L1044", "L985", "L984", "L925", "L924"],
    "characterID": ["u0", "u2", "u0", "u2", "u0", "u2"],
    "movieID": ["m0", "m0", "m0", "m0", "m0", "m0"],
    "character": ["BIANCA", "CAMERON", "BIANCA", "CAMERON", "BIANCA", "CAMERON"],
    "text": ["They do not!", "They do to!", "I hope so.", "She okay?", "Let's go.", "Wow"]
}


TITLES_DATA = "m0 +++$+++ 10 things i hate about you +++$+++ 1999 +++$+++ 6.90 +++$+++ "\
    "62847 +++$+++ ['comedy', 'romance']\n"\
    "m1 +++$+++ 1492: conquest of paradise +++$+++ 1992 +++$+++ 6.20 +++$+++ "\
    "10421 +++$+++ ['adventure', 'biography', 'drama', 'history']\n"\
    "m2 +++$+++ 15 minutes +++$+++ 2001 +++$+++ 6.10 +++$+++ 25854 +++$+++ ['act"\
    "ion', 'crime', 'drama', 'thriller']\n"\
    "m3 +++$+++ 2001: a space odyssey +++$+++ 1968 +++$+++ 8.40 +++$+++ 163227 +++$+++ "\
    "['adventure', 'mystery', 'sci-fi']"

EXPECTED_TITLES_DATA = {
    "movieID": ["m0", "m1", "m2", "m3"],
    "title": ["10 things i hate about you", "1492: conquest of paradise", "15 minutes",
              "2001: a space odyssey"],
    "year": [1999, 1992, 2001, 1968],
    "rating": [6.90, 6.20, 6.10, 8.40],
    "votes": [62847, 10421, 25854, 163227],
    "genres": [['comedy', 'romance'], ['adventure', 'biography', 'drama', 'history'],
               ['action', 'crime', 'drama', 'thriller'],
               ['adventure', 'mystery', 'sci-fi']]
}


URLS_DATA = "m0 +++$+++ 10 things i hate about you +++$+++ http://www.dailyscript.com/sc"\
    "ripts/10Things.html\n"\
    "m1 +++$+++ 1492: conquest of paradise +++$+++ http://www.hundland.org/scripts/1492-"\
    "ConquestOfParadise.txt\n"\
    "m2 +++$+++ 15 minutes +++$+++ http://www.dailyscript.com/scripts/15minutes.html\n"\
    "m3 +++$+++ 2001: a space odyssey +++$+++ http://www.scifiscripts.com/"\
    "scripts/2001.txt"

EXPECTED_URLS_DATA = {
    "movieID": ["m0", "m1", "m2", "m3"],
    "title": ["10 things i hate about you", "1492: conquest of paradise",
              "15 minutes", "2001: a space odyssey"],
    "url": ["http://www.dailyscript.com/scripts/10Things.html",
            "http://www.hundland.org/scripts/1492-ConquestOfParadise.txt",
            "http://www.dailyscript.com/scripts/15minutes.html",
            "http://www.scifiscripts.com/scripts/2001.txt"]}


def create_file(file_path, file_content):
    with open(file=file_path, mode='w', encoding="iso-8859-1") as file_p:
        file_p.write(file_content)


def create_mock_dataset(tmpdir):
    dataset_dir = os.path.join(tmpdir, CornellMovieDialogsLoader.NAME,
                               CornellMovieDialogsLoader.DATA_FOLDER_NAME)
    os.makedirs(name=dataset_dir)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.CHARACTERS_FILENAME),
        file_content=CHARACTERS_DATA)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.CONVERSATIONS_FILENAME),
        file_content=CONVERSATIONS_DATA)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.LINES_FILENAME),
        file_content=LINES_DATA)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.URL_FILENAME),
        file_content=URLS_DATA)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.TITLE_FILENAME),
        file_content=TITLES_DATA)


def test_loading_dataset(tmpdir):
    pytest.importorskip("pandas")
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", tmpdir):
        create_mock_dataset(tmpdir=tmpdir)
        data_loader = CornellMovieDialogsLoader()
        data = data_loader.load_dataset()
        assert len(data) == 5

        titles = data.titles
        assert len(titles) == len(CornellMovieDialogsLoader.TITLE_FIELDS)
        assert titles == EXPECTED_TITLES_DATA

        conversations = data.conversations
        assert len(conversations) == len(
            CornellMovieDialogsLoader.CONVERSATIONS_FIELDS)
        assert conversations == EXPECTED_CONVERSATIONS_DATA

        lines = data.lines
        assert len(lines) == len(CornellMovieDialogsLoader.LINES_FIELDS)
        assert lines == EXPECTED_LINES_DATA

        characters = data.characters
        assert len(characters) == len(CornellMovieDialogsLoader.CHARACTERS_FIELDS)
        assert characters == EXPECTED_CHARACTERS_DATA

        url = data.url
        assert len(url) == len(CornellMovieDialogsLoader.URL_FIELDS)
        assert url == EXPECTED_URLS_DATA
