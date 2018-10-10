from takepod.storage import downloader
import pytest
import tempfile
from unittest.mock import Mock
import os
import requests
from urllib3 import response

def test_base_class_abstract():
    with pytest.raises(TypeError):
        downloader.BaseDownloader()

def test_http_downloader_abstract():
    with pytest.raises(TypeError) :
        downloader.HttpDownloader()

def test_simple_url_downloader_small_file():
    URL = "https://www.hnb.hr/tecajn/htecajn.htm"
    test_string = "test string\nnewline"
    test_string_bytes = b"test string\nnewline"

    ##temporary directory
    base = tempfile.mkdtemp()
    assert os.path.exists(base)

    _mock_response(base_path = base, url = URL, data_bytes = test_string_bytes)

    ##testing download
    result_file_path = os.path.join(base, "result.txt")
    dl = downloader.SimpleHttpDownloader
    returnValue=dl.download(path=result_file_path, uri=URL)
    assert(returnValue)
    assert(os.path.exists(result_file_path))

    with open(result_file_path,'r') as result_file:
        result_content = result_file.read()
        assert(result_content==test_string) 



def test_simple_url_downloader_file_already_exists_no_overwrite():
    URL = "https://www.hnb.hr/tecajn/htecajn.htm"
    ##temporary directory
    base = tempfile.mkdtemp()
    assert os.path.exists(base)

    file_path = os.path.join(base, "file.txt")
    with open(file_path, 'w') as original_fp:
        original_fp.write("original")

    dl = downloader.SimpleHttpDownloader
    returnValue = dl.download(path=file_path, uri=URL)
    assert(returnValue == False)
    with open(file_path,'r') as original_fp:
        assert(original_fp.read()=="original")


def test_simple_url_downloader_file_already_exists_overwrite():
    URL = "https://www.hnb.hr/tecajn/htecajn.htm"
    ##temporary directory
    base = tempfile.mkdtemp()
    assert os.path.exists(base)

    file_path = os.path.join(base, "file.txt")
    with open(file_path, 'w') as original_fp:
        original_fp.write("original")

    _mock_response(base_path=base, url=URL, data_bytes=b"new")

    dl = downloader.SimpleHttpDownloader
    dl.download(path=file_path, uri=URL, overwrite=True)

    with open(file_path,'r') as original_fp:
        assert(original_fp.read()=="new")

def test_simple_url_downloader_none_path():
    URL = "https://www.hnb.hr/tecajn/htecajn.htm"
    dl = downloader.SimpleHttpDownloader
    with pytest.raises(ValueError):
        dl.download(path=None, uri= URL)

def test_simple_url_downloader_resource_not_found():
    URL = "https://www.hnb.hr/tecajn/htecajn.htm"
    ##temporary directory
    base = tempfile.mkdtemp()
    assert os.path.exists(base)

    _mock_response(base_path=base, url=URL, data_bytes=b"", status_code=404, status_reason="Not Found")

    dl = downloader.SimpleHttpDownloader
    file_path = os.path.join(base, "file.txt")
    with pytest.raises(RuntimeError):
        dl.download(path=file_path, uri=URL)

    


def _mock_response(base_path, url, data_bytes, status_code = 200, status_reason='OK'):
    ##creating expected file for mocking response reading function
    original_file_path = os.path.join(base_path, "original.txt")
    with open(original_file_path, "wb") as original_fp: 
        original_fp.write(data_bytes)

    ##mocking urllib response

    resp = requests.models.Response()
    resp.encoding = 'UTF-8'
    resp.history = '[]'
    resp.reason = status_reason
    resp.url = url
    resp.status_code = status_code
    resp._content = False
    
    original_response = response.HTTPResponse(body=resp._content, 
                                        headers=None, 
                                        status =status_code,
                                        reason=status_reason)
    response_fp = open(original_file_path, "rb")
    original_response.read = response_fp.read
    resp.raw = original_response
    requests.get = Mock(return_value=resp)