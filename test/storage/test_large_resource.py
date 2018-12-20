import os
import tempfile
import zipfile
import pytest
from takepod.storage.large_resource import LargeResource
from takepod.storage.downloader import SimpleHttpDownloader

MOCK_RESOURCE_NAME = "res"
MOCK_FILE_NAME = "test_file.txt"
MOCK_FILE_CONTENT = "content"


def create_mock_file(file_path):
    with open(file=file_path, mode="w") as fp:
        fp.write(MOCK_FILE_CONTENT)


def create_mock_zip_archive(file_name, file_path):
    with zipfile.ZipFile(file=file_path, mode="w") as zipfp:
        zipfp.writestr(zinfo_or_arcname=file_name, data=MOCK_FILE_CONTENT)


def test_arguments_url_missing():
    with pytest.raises(expected_exception=ValueError):
        LargeResource(**{LargeResource.RESOURCE_NAME: "name"})


def test_arguments_resource_name_missing():
    with pytest.raises(expected_exception=ValueError):
        LargeResource(**{LargeResource.URI: "http://fer.hr"})


def test_resource_not_archive():
    SimpleHttpDownloader.download = lambda uri, path, overwrite: \
                                create_mock_file(path)

    base = tempfile.mkdtemp()
    assert os.path.exists(base)

    LargeResource.BASE_RESOURCE_DIR = base
    LargeResource(**{LargeResource.URI: "http://fer.hr",
                     LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME})

    abs_file_path = os.path.join(base, MOCK_RESOURCE_NAME)
    assert os.path.exists(abs_file_path)
    with open(file=abs_file_path, mode='r') as fpr:
        content = fpr.read()
        assert content == MOCK_FILE_CONTENT


def test_resource_downloading_unzip():
    SimpleHttpDownloader.download = lambda uri, path, overwrite: \
                                create_mock_zip_archive(
                                    file_name=MOCK_FILE_NAME, file_path=path)

    base = tempfile.mkdtemp()
    assert os.path.exists(base)

    LargeResource.BASE_RESOURCE_DIR = base
    LargeResource(**{LargeResource.URI: "http://fer.hr",
                     LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
                     LargeResource.ARCHIVE: "zip"})

    abs_file_path = os.path.join(base, MOCK_RESOURCE_NAME, MOCK_FILE_NAME)
    assert os.path.exists(abs_file_path)
    with open(file=abs_file_path, mode='r') as fpr:
        content = fpr.read()
        assert content == MOCK_FILE_CONTENT


def test_file_zip_exists():
    SimpleHttpDownloader.download = lambda uri, path, overwrite: None

    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    os.mkdir(os.path.join(base, MOCK_RESOURCE_NAME))
    LargeResource(**{LargeResource.URI: "http://fer.hr",
                     LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
                     LargeResource.ARCHIVE: "zip"})


def test_file_not_original_archive_exists():
    SimpleHttpDownloader.download = lambda uri, path, overwrite: None

    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    os.mkdir(os.path.join(base, MOCK_RESOURCE_NAME))
    LargeResource(**{LargeResource.URI: "http://fer.hr",
                     LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
                     })


def test_unsupported_archive_type():
    SimpleHttpDownloader.download = lambda uri, path, overwrite: \
                                create_mock_zip_archive(
                                    file_name=MOCK_FILE_NAME, file_path=path)

    base = tempfile.mkdtemp()
    assert os.path.exists(base)

    LargeResource.BASE_RESOURCE_DIR = base

    with pytest.raises(ValueError):
        LargeResource(**{LargeResource.URI: "http://fer.hr",
                         LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
                         LargeResource.ARCHIVE: "archive_not_supp"})
