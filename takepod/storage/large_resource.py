"""Module contains class for defining large resource. Classes that contain
large resources that should be downloaded should use this module."""
import os
import tempfile
from takepod.storage.downloader import SimpleHttpDownloader
from takepod.storage import utility


class LargeResource:
    """Large resource that needs to download files from URL. Class also
    supports archive decompression.

    Attributes
    ----------
    BASE_RESOURCE_DIR : str
        base large files directory path
    RESOURCE_NAME : str
        key for defining resource directory name parameter
    URL : str
        key for defining resource url parameter
    ARCHIVE : str
        key for defining archiving method paramter
    SUPPORTED_ARCHIVE : list(str)
        list of supported archive file types
    """
    BASE_RESOURCE_DIR = "."
    RESOURCE_NAME = "resource"
    URL = "url"
    ARCHIVE = "archive"
    SUPPORTED_ARCHIVE = ["zip", "tar", "bz2", "lzma"]

    def __init__(self, **kwargs):
        """Creates large resource file. If the file is not in resource_location
        it will be dowloaded from url and if needed decompressed.
        Resource location is defined as BASE_RESOURCE_DIR+RESOURCE_NAME

        Parameters
        ----------
        kwargs : dict(str, str)
            key word arguments that define RESOURCE_NAME, URL and optionally
            archiving method ARCHIVE

        """
        self._check_args(arguments=kwargs)
        self.config = kwargs
        self.resource_location = os.path.join(
            LargeResource.BASE_RESOURCE_DIR,
            self.config[LargeResource.RESOURCE_NAME])
        self._check_files()

    def _check_files(self):
        """Method checks if large resource files exists and if they don't it
        initiates downloading of resources."""
        if os.path.exists(self.resource_location):
            return
        if LargeResource.ARCHIVE in self.config\
            and self.config[LargeResource.ARCHIVE]:
            self._download_unarchive()
            return
        self._download(download_destination=self.resource_location)

    def _download(self, download_destination):
        """Method downloades file from config URL to given directory.

        Parameters
        ----------
        download_destination : str
            place where to download resource
        """
        SimpleHttpDownloader.download(uri=self.config[LargeResource.URL],
                                      path=download_destination,
                                      overwrite=False)

    def _unarchive(self, archive_file):
        """Method unarchives given archive file if decompression of given file
        is supported.

        Parameters
        ----------
        archive_file : str
            path to archive file that needs to be decompressed

        Raises
        ------
        ValueError
            if configured archiving method is not supported
        """
        if self.config[LargeResource.ARCHIVE] \
                not in LargeResource.SUPPORTED_ARCHIVE:
            raise ValueError("Unsupported archive method. Given {}, expected"
                             "one from {}".format(
                                 self.config[LargeResource.ARCHIVE],
                                 LargeResource.SUPPORTED_ARCHIVE))
        if self.config[LargeResource.ARCHIVE] == "zip":
            utility.extract_zip_file(archive_file=archive_file,
                                     destination_dir=self.resource_location)
            return
        utility.extract_tar_file(archive_file=archive_file,
                                 destination_dir=self.resource_location)

    def _download_unarchive(self):
        """Method downloades resource and decompresses it to resource location.
        """
        os.makedirs(name=self.resource_location)
        download_dir = os.path.join(tempfile.mkdtemp(),
                                    self.config[LargeResource.RESOURCE_NAME])
        self._download(download_destination=download_dir)
        self._unarchive(archive_file=download_dir)

    def _check_args(self, arguments):
        """Method checks if the large resource configuration has all essential
        parts such as large resource url and resource name.

        Parameters
        ----------
        arguments : dict(str, str)
            dictionary containing large resource configuration

        Raises
        ------
        ValueError
            if resource name or url are not defined
        """
        essential_arguments = [LargeResource.RESOURCE_NAME, LargeResource.URL]
        for arg in essential_arguments:
            if arg not in arguments or not arguments[arg]:
                raise ValueError(arg+" must be defined"
                                 " while defining Large Resource")