from abc import ABC, abstractclassmethod
import shutil
import os
import requests

class BaseDownloader(ABC):
    '''BaseDownloader interface for downloader classes.'''
    @abstractclassmethod
    def download(cls, uri, path, overwrite=False):
        '''Function downloades file from given URI to given path. If the overwrite variable is true
       and given path already exists it will be overwriten with new file.

        Parameters
        ----------
        uri : str
              URI of file that needs to be downloaded
        path : str
              destination path where to save downloaded file
        overwrite : bool
                    if true and given path exists downloaded file will overwrite existing files

            :raise ValueError: if given uri or path are None
            :raise RuntimeError: if there was an error while obtaining resource from uri



        '''
        pass

class HttpDownloader(BaseDownloader, ABC):
    '''Interface for downloader that uses http protocol for data transfer.'''
    @classmethod
    def _process_response(cls, response, output_file, chuck_length=1024*16):
        """Function process given HTTP response and copies it to the given outputfile.
            Data is processed in chunks of given length.
            :param response: HTTP response obtained by requests.get method. 
                             Response should be streamed. 
            :param output_file: file like object where to copy response content
            :raise ValueError: if given response or output_file are None
            :raise RuntimeError: if given HTTP response wasn't successful (response code >= 300)
        """
        if response is None or output_file is None:
            raise ValueError("Response object and output file object mustn't be None.")
        if response.status_code >= 300:
            raise RuntimeError("Given file is not accessible. HTTP response code "
                               +str(response.status_code)+
                               " "+str(response.reason))
        shutil.copyfileobj(response.raw, output_file)
        return True

class SimpleHttpDownloader(HttpDownloader):
    '''Downloader that uses HTTP protocol for downloading. It doesn't offer content confirmation 
    (as needed for example in google drive) or any kind of authentication.'''
    @classmethod
    def download(cls, uri, path, overwrite=False):
        if path is None or uri is None:
            raise ValueError("Path and url mustn't be None. Given path: "
                             +str(path)+", url: "+str(uri))
        if not overwrite and os.path.exists(path):
            return False
        with requests.get(uri, headers={'User-Agent': 'Mozilla/5.0'}, stream=True) as response,\
             open(path, 'wb') as output_file:
            success = cls._process_response(response, output_file)
            return success
            