from takepod.storage import downloader
import pytest
import tempfile
from unittest.mock import Mock
import os

def test_base_class_abstract():
    with pytest.raises(TypeError) as excinfo:
        downloader.BaseDownloader()

def test_http_downloader_abstract():
    with pytest.raises(TypeError) as excinfo:
        downloader.HttpDownloader()

def test_simple_url_downloader_small_file():
    URL = "https://www.hnb.hr/tecajn/htecajn.htm"
    test_string = "test string\nnewline"
    test_string_bytes = b"test string\nnewline"

    ##temporary directory
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    
    ##creating expected file for mocking response reading function
    original_file_path = os.path.join(base, "original.txt")
    with open(original_file_path, "wb") as original_fp: 
        original_fp.write(test_string_bytes)

    ##mocking urllib response
    import requests
    from urllib3 import response
    resp = requests.models.Response()
    resp.encoding = 'UTF-8'
    resp.history = '[]'
    resp.reason = 'OK'
    resp.url = 'https://www.hnb.hr/tecajn/htecajn.htm'
    resp.status_code = 200
    resp._content = False
    
    original_response = response.HTTPResponse(body=resp._content, headers=None, status = 200, reason='OK')
    response_fp = open(original_file_path, "rb")
    original_response.read = response_fp.read
    resp.raw = original_response
    requests.get = Mock(return_value=resp)

    ##testing download
    result_file_path = os.path.join(base, "result.txt")
    dl = downloader.SimpleHttpDownloader
    dl.download(path=result_file_path, url=URL)

    assert(os.path.exists(result_file_path))

    with open(result_file_path) as result_file:
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
    dl.download(path=file_path, url=URL)

    with open(file_path,'r') as original_fp:
        assert(original_fp.read()=="original")

