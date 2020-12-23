import os
import zipfile

import paramiko
import pytest

from podium.storage.resources.downloader import SCPDownloader, SimpleHttpDownloader
from podium.storage.resources.large_resource import LargeResource, SCPLargeResource


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


def test_resource_not_archive(tmpdir):
    def download(uri, path, overwrite):
        create_mock_file(path)

    SimpleHttpDownloader.download = download

    LargeResource.BASE_RESOURCE_DIR = tmpdir
    LargeResource(
        **{
            LargeResource.URI: "http://fer.hr",
            LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
        }
    )

    abs_file_path = os.path.join(tmpdir, MOCK_RESOURCE_NAME)
    assert os.path.exists(abs_file_path)
    with open(file=abs_file_path, mode="r") as fpr:
        content = fpr.read()
        assert content == MOCK_FILE_CONTENT


def test_resource_downloading_unzip(tmpdir):
    def download(uri, path, overwrite):
        create_mock_zip_archive(file_name=MOCK_FILE_NAME, file_path=path)

    SimpleHttpDownloader.download = download

    LargeResource.BASE_RESOURCE_DIR = tmpdir
    LargeResource(
        **{
            LargeResource.URI: "http://fer.hr",
            LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
            LargeResource.ARCHIVE: "zip",
        }
    )

    abs_file_path = os.path.join(tmpdir, MOCK_RESOURCE_NAME, MOCK_FILE_NAME)
    assert os.path.exists(abs_file_path)
    with open(file=abs_file_path, mode="r") as fpr:
        content = fpr.read()
        assert content == MOCK_FILE_CONTENT


def test_file_zip_exists(tmpdir):
    SimpleHttpDownloader.download = lambda uri, path, overwrite: None

    os.mkdir(os.path.join(tmpdir, MOCK_RESOURCE_NAME))
    LargeResource.BASE_RESOURCE_DIR = tmpdir
    LargeResource(
        **{
            LargeResource.URI: "http://fer.hr",
            LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
            LargeResource.ARCHIVE: "zip",
        }
    )


def test_file_not_original_archive_exists(tmpdir):
    SimpleHttpDownloader.download = lambda uri, path, overwrite: None

    os.mkdir(os.path.join(tmpdir, MOCK_RESOURCE_NAME))
    LargeResource(
        **{
            LargeResource.URI: "http://fer.hr",
            LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
        }
    )


def test_unsupported_archive_type(tmpdir):
    def download(uri, path, overwrite):
        create_mock_zip_archive(file_name=MOCK_FILE_NAME, file_path=path)

    SimpleHttpDownloader.download = download

    LargeResource.BASE_RESOURCE_DIR = tmpdir

    with pytest.raises(ValueError):
        LargeResource(
            **{
                LargeResource.URI: "http://fer.hr",
                LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
                LargeResource.ARCHIVE: "archive_not_supp",
            }
        )


def test_scp_download_file(tmpdir):
    def download(uri, path, overwrite, **kwargs):
        create_mock_file(path)

    SCPDownloader.download = download

    LargeResource.BASE_RESOURCE_DIR = tmpdir

    SCPLargeResource(
        **{
            LargeResource.URI: "http://fer.hr",
            LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
            SCPLargeResource.SCP_HOST_KEY: "djurdja.fer.hr",
            SCPLargeResource.SCP_USER_KEY: "user",
            SCPLargeResource.SCP_PASS_KEY: "password",
            SCPLargeResource.SCP_PRIVATE_KEY: "D:\\TakeLab\\" "takleab_ssh",
        }
    )

    abs_file_path = os.path.join(tmpdir, MOCK_RESOURCE_NAME)
    assert os.path.exists(abs_file_path)
    with open(file=abs_file_path, mode="r") as fpr:
        content = fpr.read()
        assert content == MOCK_FILE_CONTENT


def test_scp_download_file_paraminko_mock(tmpdir):
    paramiko.SSHClient.connect = lambda **kwards: None
    paramiko.SFTPClient.get = lambda remotepath, localpath: create_mock_file(remotepath)

    LargeResource.BASE_RESOURCE_DIR = tmpdir

    SCPLargeResource(
        **{
            LargeResource.URI: "http://fer.hr",
            LargeResource.RESOURCE_NAME: MOCK_RESOURCE_NAME,
            SCPLargeResource.SCP_HOST_KEY: "djurdja.fer.hr",
            SCPLargeResource.SCP_USER_KEY: "user",
            SCPLargeResource.SCP_PASS_KEY: "password",
            SCPLargeResource.SCP_PRIVATE_KEY: "D:\\TakeLab\\" "takleab_ssh",
        }
    )

    abs_file_path = os.path.join(tmpdir, MOCK_RESOURCE_NAME)
    assert os.path.exists(abs_file_path)
    with open(file=abs_file_path, mode="r") as fpr:
        content = fpr.read()
        assert content == MOCK_FILE_CONTENT
