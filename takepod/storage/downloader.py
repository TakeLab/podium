from abc import ABC,abstractclassmethod
import requests
import shutil
import os


class BaseDownloader(ABC):
    '''BaseDownloader interface for downloader classes.'''
    @abstractclassmethod
    def download (cls, uri, path, overwrite=False):
        '''Function downloades file from given URI to given path. If the overwrite variable is true and given path
         already exists it will be overwriten with new file.
            :param uri: URI of file that needs to be downloaded
            :type uri: str
            :param path: destination path where to save downloaded file
            :type path: str
            :param overwrite: if true and given path exists downloaded file will overwrite existing files
            :type overwrite: bool
            :raise ValueError: if given uri or path are None
            :raise RuntimeError: if there was an error while obtaining resource from uri

        '''
        pass

class HttpDownloader(BaseDownloader, ABC):
    @classmethod
    def _process_response(cls, response, output_file):
        """Function process given HTTP response and copies it to the given outputfile.  
            :param response: HTTP response obtained by requests.get method. Response should be streamed. 
            :param output_file: file like object where to copy response content
            :raise ValueError: if given response or output_file are None
            :raise RuntimeError: if given HTTP response wasn't successful (response code >= 300)
        """
        if (response==None or output_file==None):
            raise ValueError("Response object and output file object mustn't be None.")
        if (response.status_code >= 300):
            raise RuntimeError("Given file is not accessible. HTTP response code "+str(response.status_code)+
            " "+str(response.reason))
        shutil.copyfileobj(response.raw, output_file)
        return True

class SimpleHttpDownloader(HttpDownloader):
    @classmethod
    def download(cls, uri, path, overwrite=False):
        if (path == None or uri == None):
            raise ValueError("Path and url mustn't be None. Given path: "+str(path)+", url: "+str(uri))
        if not overwrite and os.path.exists(path):
            return False
        with requests.get(uri, headers={'User-Agent': 'Mozilla/5.0'}, stream=True) as response,\
             open(path, 'wb') as output_file:
            success = cls._process_response(response, output_file)
            return success
            