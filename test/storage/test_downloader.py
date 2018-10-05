from takepod.storage import downloader
import pytest
import requests
import requests.models as reqmod
import tempfile
import os
from urllib3.response import HTTPResponse

def test_base_class_abstract():
    with pytest.raises(TypeError) as excinfo:
        downloader.BaseDownloader()

def test_http_downloader_abstract():
    with pytest.raises(TypeError) as excinfo:
        downloader.HttpDownloader()

def test_http_downloader_get_google_drive():
    url = 'https://drive.google.com/open?id=1r86pGwqyEKcqBQFruyN0U0AblnOlrHo-'
    dl = downloader.HttpDownloader.getDownloader(url)
    assert(dl == downloader.GoogleDriveDownloader)

def test_http_downloader_get_simple_http_downloader():
    url = 'https://www.hnb.hr/tecajn/htecajn.htm'
    dl = downloader.HttpDownloader.getDownloader(url)
    assert(dl == downloader.SimpleHttpDownloader)


""" def test_http_downloader_process_response():
    url = 'https://www.hnb.hr/tecajn/htecajn.htm'
    dl = downloader.HttpDownloader.getDownloader(url)

    base = tempfile.mkdtemp()
    assert (os.path.exists(base))
    
    path = os.path.join(base, 'file.txt')

    response = generate_response()
    dl._process_response(path=path, http_response=response)
    assert(os.path.exists(path))
    with open(path, 'rb') as fp:
        filecontent = fp.read()
        assert(filecontent == response._content)
     """

## TODO add response mock object so that process response can be tested

def test_http_downloader_process_response_not_found():
    pass

def generate_response():
    pass  
    """ resp = reqmod.Response()
    resp.encoding = 'UTF-8'
    resp.history = '[]'
    resp.reason = 'OK'
    resp.url = 'https://www.hnb.hr/tecajn/htecajn.htm'
    resp.status_code = 200
    resp._content = False
    
    original_response = HTTPResponse(body=resp._content, headers=None, status = 200, reason='OK')
    original_response.headers = {'X-OneAgent-JS-Injection': 'true', 'Set-Cookie': 'dtCookie=BD158C3E0A455C4A4CDCE70C16E5A0D6||1; Domain=.hnb.hr; Path=/, HNB_cookie=rd30o00000000000000000000ffffc0a80751o8443; path=/; Httponly; Secure, TS013545bc=01c7caa5c42c5b48ef1bb214ec5ac780a5b52d98552c0d23d7b66873e657ca3eb7a6ed301bf47388e3ff0e00091873f13963c29049a57c6d39eeff6adb3e23e69d608e6534aa424181c5bb1a05ce1a4963fffa2146; Path=/; Domain=.www.hnb.hr', 'X-Content-Type-Options': 'nosniff',
'X-Frame-Options': 'SAMEORIGIN', 'X-XSS-Protection': '1', 'Content-Type': 'text/plain;charset=UTF-8', 'Date': 'Fri, 05 Oct 2018 15:15:29 GMT', 'Transfer-Encoding': 'chunked'}
    resp.raw=original_response
    
    return resp
 """
