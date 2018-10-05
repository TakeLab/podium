from abc import ABC,abstractclassmethod
import requests
from tqdm import tqdm

class BaseDownloader(ABC):
    @abstractclassmethod
    def download (cls, url, path):
        pass

class HttpDownloader(BaseDownloader, ABC):
    @classmethod
    def _process_response(cls, path, http_response):
        chunk_size = 16 * 1024
        total_size = int(http_response.headers.get('Content-length', 0))
        with open(path, "wb") as file:
            with tqdm(total=total_size, unit='B',
                      unit_scale=1, desc=path.split('/')[-1]) as t:
                for chunk in http_response.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))
    @staticmethod
    def getDownloader(url):
        if 'drive.google.com' in url:
            return GoogleDriveDownloader
        else:
            return UrlDownloader

class UrlDownloader(HttpDownloader):
    @classmethod
    def download(cls, url, path):
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        cls._process_response(path, response)

class GoogleDriveDownloader(HttpDownloader):
    @classmethod
    def download(cls, url, path):
        if 'drive.google.com' not in url:
            raise ValueError("not supported url type")
        confirm_token = None
        session = requests.Session()
        response = session.get(url, stream=True)
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                confirm_token = v

        if confirm_token:
            url = url + "&confirm=" + confirm_token
            response = session.get(url, stream=True)

        cls._process_response(path, response)

    