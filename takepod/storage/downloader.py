from abc import ABC,abstractclassmethod
import requests
import shutil
import os


class BaseDownloader(ABC):
    @abstractclassmethod
    def download (cls, url, path, overwrite=False):
        pass

class HttpDownloader(BaseDownloader, ABC):
    @classmethod
    def _process_response(cls, response, output_file):
        if (response==None or output_file==None):
            raise ValueError("Response object and output file object mustn't be None.")
        if (response.status_code >= 300):
            raise RuntimeError("Given file is not accessible. HTTP response code "+str(response.status_code)+" "+str(response.reason))
        shutil.copyfileobj(response.raw, output_file)
        return True

class SimpleHttpDownloader(HttpDownloader):
    @classmethod
    def download(cls, url, path, overwrite=False):
        if not overwrite and os.path.exists(path):
            return False
        with requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True) as response, open(path, 'wb') as output_file:
            success = cls._process_response(response, output_file)
            return success
            