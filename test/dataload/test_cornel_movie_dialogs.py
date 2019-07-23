import os
from mock import patch
import pytest
from takepod.storage import LargeResource
from takepod.dataload.cornel_movie_dialogs import CornellMovieDialogsLoader

CHARCTERS_DATA = "u0 +++$+++ BIANCA +++$+++ m0 +++$+++ 10 things i hate about you +++$++"\
    "+ f +++$+++ 4\nu1 +++$+++ BRUCE +++$+++ m0 +++$+++ 10 things i hate about you +++$+"\
    "++ ? +++$+++ ?\nu2 +++$+++ CAMERON +++$+++ m0 +++$+++ 10 things i hate about you ++"\
    "+$+++ m +++$+++ 3\nu3 +++$+++ CHASTITY +++$+++ m0 +++$+++ 10 things i hate about yo"\
    "u +++$+++ ? +++$+++ ?"


CONVERSATIONS_DATA = "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']"\
    "\nu0 +++$+++ u2 +++$+++ m0 +++$+++ ['L198', 'L199']\nu0 +++$+++ u2 +++$+++ m0 +++$"\
    "+++ ['L200', 'L201', 'L202', 'L203']\nu0 +++$+++ u2 +++$+++ m0 +++$+++ ['L204', 'L"\
    "205', 'L206']\nu0 +++$+++ u2 +++$+++ m0 +++$+++ ['L207', 'L208']"


LINES_DATA = "L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!\nL1044 +++"\
    "$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!\nL985 +++$+++ u0 +++$+++ m0 "\
    "+++$+++ BIANCA +++$+++ I hope so.\nL984 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+"\
    "++ She okay?\nL925 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Let's go.\nL924 +++"\
    "$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Wow"


TITLES_DATA = "m0 +++$+++ 10 things i hate about you +++$+++ 1999 +++$+++ 6.90 +++$+++ "\
    "62847 +++$+++ ['comedy', 'romance']\nm1 +++$+++ 1492: conquest of paradise +++$+++"\
    " 1992 +++$+++ 6.20 +++$+++ 10421 +++$+++ ['adventure', 'biography', 'drama', 'hist"\
    "ory']\nm2 +++$+++ 15 minutes +++$+++ 2001 +++$+++ 6.10 +++$+++ 25854 +++$+++ ['act"\
    "ion', 'crime', 'drama', 'thriller']\nm3 +++$+++ 2001: a space odyssey +++$+++ 1968"\
    " +++$+++ 8.40 +++$+++ 163227 +++$+++ ['adventure', 'mystery', 'sci-fi']"


URLS_DATA = "m0 +++$+++ 10 things i hate about you +++$+++ http://www.dailyscript.com/sc"\
    "ripts/10Things.html\nm1 +++$+++ 1492: conquest of paradise +++$+++ http://www.hundl"\
    "and.org/scripts/1492-ConquestOfParadise.txt\nm2 +++$+++ 15 minutes +++$+++ http://w"\
    "ww.dailyscript.com/scripts/15minutes.html\nm3 +++$+++ 2001: a space odyssey +++$+++"\
    " http://www.scifiscripts.com/scripts/2001.txt"


def create_file(file_path, file_content):
    with open(file=file_path, mode='w', encoding="iso-8859-1") as file_p:
        file_p.write(file_content)


def create_mock_dataset(tmpdir):
    dataset_dir = os.path.join(tmpdir, CornellMovieDialogsLoader.NAME,
                               CornellMovieDialogsLoader.DATA_FOLDER_NAME)
    os.makedirs(name=dataset_dir)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.MOVIE_CHARACTERS_FILENAME),
        file_content=CHARCTERS_DATA)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.MOVIE_CONVERSATIONS_FILENAME),
        file_content=CONVERSATIONS_DATA)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.MOVIE_LINES_FILENAME),
        file_content=LINES_DATA)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.MOVIE_URL_FILENAME),
        file_content=URLS_DATA)
    create_file(
        file_path=os.path.join(dataset_dir,
                               CornellMovieDialogsLoader.MOVIE_TITLE_FILENAME),
        file_content=TITLES_DATA)


def test_loading_dataset(tmpdir):
    pytest.importorskip("pandas")
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", tmpdir):
        create_mock_dataset(tmpdir=tmpdir)
        data_loader = CornellMovieDialogsLoader()
        data = data_loader.load_dataset()
        assert len(data) == 5

        titles = data.titles
        assert len(titles) == len(CornellMovieDialogsLoader.MOVIE_TITLE_FIELDS)

        conversations = data.conversations
        assert len(conversations) == len(
            CornellMovieDialogsLoader.MOVIE_CONVERSATIONS_FIELDS)

        lines = data.lines
        assert len(lines) == len(CornellMovieDialogsLoader.MOVIE_LINES_FIELDS)

        characters = data.characters
        assert len(characters) == len(CornellMovieDialogsLoader.MOVIE_CHARACTERS_FIELDS)

        url = data.url
        assert len(url) == len(CornellMovieDialogsLoader.MOVIE_URL_FIELDS)
