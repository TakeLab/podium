import os
from mock import patch
import pytest
from takepod.storage import LargeResource
from takepod.dataload.cornel_movie_dialogs import CornellMovieDialogsLoader

CHARACTERS_DATA = "u0 +++$+++ BIANCA +++$+++ m0 +++$+++ 10 things i hate about you "\
    "+++$+++ f +++$+++ 4\n"\
    "u1 +++$+++ BRUCE +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ ? +++$+++ ?"\
    "\n"\
    "u2 +++$+++ CAMERON +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ m +++$+++"\
    " 3\n"\
    "u3 +++$+++ CHASTITY +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ ? +++$+++"\
    " ?"


CONVERSATIONS_DATA = "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']"\
    "\n"\
    "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L198', 'L199']\n"\
    "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L200', 'L201', 'L202', 'L203']\n"\
    "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L204', 'L205', 'L206']\n"\
    "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L207', 'L208']"


LINES_DATA = "L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!\n"\
    "L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!\n"\
    "L985 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I hope so.\n"\
    "L984 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ She okay?\n"\
    "L925 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Let's go.\n"\
    "L924 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Wow"


TITLES_DATA = "m0 +++$+++ 10 things i hate about you +++$+++ 1999 +++$+++ 6.90 +++$+++ "\
    "62847 +++$+++ ['comedy', 'romance']\n"\
    "m1 +++$+++ 1492: conquest of paradise +++$+++ 1992 +++$+++ 6.20 +++$+++ "\
    "10421 +++$+++ ['adventure', 'biography', 'drama', 'history']\n"\
    "m2 +++$+++ 15 minutes +++$+++ 2001 +++$+++ 6.10 +++$+++ 25854 +++$+++ ['act"\
    "ion', 'crime', 'drama', 'thriller']\n"\
    "m3 +++$+++ 2001: a space odyssey +++$+++ 1968 +++$+++ 8.40 +++$+++ 163227 +++$+++ "\
    "['adventure', 'mystery', 'sci-fi']"


URLS_DATA = "m0 +++$+++ 10 things i hate about you +++$+++ http://www.dailyscript.com/sc"\
    "ripts/10Things.html\n"\
    "m1 +++$+++ 1492: conquest of paradise +++$+++ http://www.hundland.org/scripts/1492-"\
    "ConquestOfParadise.txt\n"\
    "m2 +++$+++ 15 minutes +++$+++ http://www.dailyscript.com/scripts/15minutes.html\n"\
    "m3 +++$+++ 2001: a space odyssey +++$+++ http://www.scifiscripts.com/"\
    "scripts/2001.txt"


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

        conversations = data.conversations
        assert len(conversations) == len(
            CornellMovieDialogsLoader.CONVERSATIONS_FIELDS)

        lines = data.lines
        assert len(lines) == len(CornellMovieDialogsLoader.LINES_FIELDS)

        characters = data.characters
        assert len(characters) == len(CornellMovieDialogsLoader.CHARACTERS_FIELDS)

        url = data.url
        assert len(url) == len(CornellMovieDialogsLoader.URL_FIELDS)
