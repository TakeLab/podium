"""Module downloader offers classes for downloading files from the given uri.
It is consisted from base class BaseDownloader that every downloader
implements. Special class of downloaders are downloaders that use HTTP
protocol, their base class is HTTPDownloader and its simple implementation
is SimpleHttpDownloader.

"""
from abc import ABC, abstractclassmethod
import os
import requests
import paramiko
from takepod.storage.utility import copyfileobj_with_tqdm



class BaseDownloader(ABC):
    '''BaseDownloader interface for downloader classes.'''
    @abstractclassmethod
    def download(cls, uri, path, overwrite=False, **kwargs):
        '''Function downloades file from given URI to given path.
        If the overwrite variable is true and given path already exists
        it will be overwriten with new file.

        Parameters
        ----------
        uri : str
              URI of file that needs to be downloaded
        path : str
              destination path where to save downloaded file
        overwrite : bool
                    if true and given path exists downloaded file
                    will overwrite existing files
        Returns
        -------
        rewrite_status: bool
            True if download was successful or False if the file already exists
            and given overwrite value was False.
        Raises
        ------
        ValueError
            if given uri or path are None
        RuntimeError
            if there was an error while obtaining resource from uri

        '''
        pass


class SCPDownloader(BaseDownloader):
    USER_NAME_KEY = "username"
    PASSWORD_KEY = "password"
    HOST_ADDR_KEY = "host"
    PUBLIC_KEY_FILE_KEY = "pub_key"
    @classmethod
    def download(cls, uri, path, overwrite=False, **kwargs):
        if path is None or uri is None:
            raise ValueError(
                "Path and url mustn't be None."
                "Given path: {}, {}".format(str(path), str(uri)))

        if not overwrite and os.path.exists(path):
            return False

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if cls.PUBLIC_KEY_FILE_KEY not in kwargs:
            kwargs[cls.PUBLIC_KEY_FILE_KEY] = None
        if cls.PASSWORD_KEY not in kwargs:
            kwargs[cls.PASSWORD_KEY] = None
        print(kwargs[cls.PUBLIC_KEY_FILE_KEY])
        client.connect(hostname=kwargs[cls.HOST_ADDR_KEY],
                       username=kwargs[cls.USER_NAME_KEY],
                       password=kwargs[cls.PASSWORD_KEY],
                       key_filename=kwargs[cls.PUBLIC_KEY_FILE_KEY])
        sftp = client.open_sftp()
        sftp.get(localpath=path, remotepath=uri)
        sftp.close()
        client.close()
        return True

class HttpDownloader(BaseDownloader, ABC):
    '''Interface for downloader that uses http protocol for data transfer.'''
    @classmethod
    def _process_response(cls, response, output_file, chuck_length=1024*16):
        """Function process given HTTP response and copies it to the given
        outputfile. Data is processed in chunks of given length.
        Parameters
        ----------
        response :
            HTTP response obtained by requests.get method.
            Response should be streamed.
        output_file :
            file like object where to copy response content
        chunck_lenght : int
            buffer size used while copying response to the output_file,
            default value is taken from shutil.copyfileobj
        Raises
        ------
        ValueError
            if given response or output_file are None
        RuntimeError
            if given HTTP response wasn't successful (response code >= 300)

        """
        if response is None or output_file is None:
            raise ValueError("Response object and output file object mustn't"
                             " be None.")
        if response.status_code >= 300:
            raise RuntimeError("Given file is not accessible because {},"
                               "HTTP response code {}"
                               .format(response.reason, response.status_code))
        copyfileobj_with_tqdm(response.raw, output_file,
                              total_size=int(response.headers.get(
                                  'Content-length', 0)),
                              buffer_size=chuck_length)
        return True


class SimpleHttpDownloader(HttpDownloader):
    '''Downloader that uses HTTP protocol for downloading. It doesn't offer
    content confirmation (as needed for example in google drive)
    or any kind of authentication.

    '''
    @classmethod
    def download(cls, uri, path, overwrite=False, **kwargs):
        if path is None or uri is None:
            raise ValueError(
                "Path and url mustn't be None."
                "Given path: {}, {}".format(str(path), str(uri)))
        if not overwrite and os.path.exists(path):
            return False

        with requests.get(url=uri, headers={'User-Agent': 'Mozilla/5.0'},
                          stream=True) as response, \
                open(path, 'wb') as output_file:
            success = cls._process_response(response, output_file)
            return success


if __name__ == "__main__":
    config = {SCPDownloader.HOST_ADDR_KEY:"djurdja.takelab.fer.hr",
              SCPDownloader.USER_NAME_KEY:"user",
              SCPDownloader.PASSWORD_KEY:"password",
              SCPDownloader.PUBLIC_KEY_FILE_KEY:"D:\\TakeLab\\"
                                                "takleab_ssh"}
    SCPDownloader.download(uri="/home/dpluscec/test_file.txt",
                           path="test_file.txt", **config)
