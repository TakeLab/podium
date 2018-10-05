from takepod.storage import downloader
import pytest

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
