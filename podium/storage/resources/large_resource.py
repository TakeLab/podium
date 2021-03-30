"""
Module contains class for defining large resource.

Classes that contain large resources that should be downloaded should use this
module.
"""
import getpass
import logging
import os
import tempfile

from podium.storage.resources import util
from podium.storage.resources.downloader import SCPDownloader, SimpleHttpDownloader


class LargeResource:
    """
    Large resource that needs to download files from URL. Class also supports
    archive decompression.

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
    URI = "uri"
    ARCHIVE = "archive"
    SUPPORTED_ARCHIVE = ["zip", "tar", "bz2", "lzma"]

    def __init__(self, **kwargs):
        """
        Creates large resource file. If the file is not in resource_location it
        will be dowloaded from url and if needed decompressed. Resource location
        is defined as BASE_RESOURCE_DIR+RESOURCE_NAME.

        Parameters
        ----------
        kwargs : dict(str, str)
            key word arguments that define RESOURCE_NAME, URL and optionally
            archiving method ARCHIVE
        """
        self._check_args(arguments=kwargs)
        self.config = kwargs
        self.resource_location = os.path.join(
            LargeResource.BASE_RESOURCE_DIR, self.config[LargeResource.RESOURCE_NAME]
        )
        self._check_files()
        logging.info(
            "Large resource %s initialized.", self.config[LargeResource.RESOURCE_NAME]
        )

    def _check_files(self):
        """
        Method checks if large resource files exists and if they don't it
        initiates downloading of resources.
        """
        if os.path.exists(self.resource_location):
            logging.info("Large resource alreadys exists, skipping download.")
            return
        logging.info("Large resource doesn't exist, starting download.")
        if LargeResource.ARCHIVE in self.config and self.config[LargeResource.ARCHIVE]:
            self._download_unarchive()
            return
        self._download(download_destination=self.resource_location)

    def _download(self, download_destination):
        """
        Method downloades file from config URL to given directory.

        Parameters
        ----------
        download_destination : str
            place where to download resource
        """
        SimpleHttpDownloader.download(
            uri=self.config[LargeResource.URI], path=download_destination, overwrite=False
        )

    def _unarchive(self, archive_file):
        """
        Method unarchives given archive file if decompression of given file is
        supported.

        Parameters
        ----------
        archive_file : str
            path to archive file that needs to be decompressed

        Raises
        ------
        ValueError
            If configured archiving method is not supported.
        """
        if self.config[LargeResource.ARCHIVE] not in LargeResource.SUPPORTED_ARCHIVE:
            raise ValueError(
                "Unsupported archive method. "
                f"Given {self.config[LargeResource.ARCHIVE]}, expected one "
                f"from {LargeResource.SUPPORTED_ARCHIVE}"
            )
        if self.config[LargeResource.ARCHIVE] == "zip":
            util.extract_zip_file(
                archive_file=archive_file, destination_dir=self.resource_location
            )
            return
        util.extract_tar_file(
            archive_file=archive_file, destination_dir=self.resource_location
        )

    def _download_unarchive(self):
        """
        Method downloades resource and decompresses it to resource location.
        """
        os.makedirs(name=self.resource_location)
        with tempfile.TemporaryDirectory() as temp_dir:
            download_dir = os.path.join(
                temp_dir, self.config[LargeResource.RESOURCE_NAME]
            )
            self._download(download_destination=download_dir)
            self._unarchive(archive_file=download_dir)

    def _check_args(self, arguments):
        """
        Method checks if the large resource configuration has all essential
        parts such as large resource url and resource name.

        Parameters
        ----------
        arguments : dict(str, str)
            dictionary containing large resource configuration

        Raises
        ------
        ValueError
            If resource name or url are not defined.
        """
        essential_arguments = [LargeResource.RESOURCE_NAME, LargeResource.URI]
        for arg in essential_arguments:
            if arg not in arguments or not arguments[arg]:
                raise ValueError(f"Large resource argument {arg} is missing.")


class SCPLargeResource(LargeResource):
    """
    Large resource that needs to download files from URI using scp protocol. For
    other functionalities class uses Large Resource class.

    Attributes
    ----------
    SCP_HOST_KEY : str
        key for keyword argument that defines remote host address
    SCP_USER_KEY : str
        key for keyword argument that defines remote host username
    SCP_PASS_KEY : str, optional
        key for keyword argument that defines remote host password or
        passphrase used in private key
    SCP_PRIVATE_KEY : str, optional
        key for keyword argument that defines location for private key
        on linux OS it can be optional if the key is in default location
    """

    SCP_HOST_KEY = "scp_host"
    SCP_USER_KEY = "scp_user"
    SCP_PASS_KEY = "scp_pass"
    SCP_PRIVATE_KEY = "scp_priv"

    def __init__(self, **kwargs):
        self._scp_config = {
            SCPDownloader.HOST_ADDR_KEY: kwargs.get(SCPLargeResource.SCP_HOST_KEY),
            SCPDownloader.USER_NAME_KEY: kwargs.get(SCPLargeResource.SCP_USER_KEY),
            SCPDownloader.PASSWORD_KEY: kwargs.get(SCPLargeResource.SCP_PASS_KEY),
            SCPDownloader.PRIVATE_KEY_FILE_KEY: kwargs.get(
                SCPLargeResource.SCP_PRIVATE_KEY
            ),
        }
        super(SCPLargeResource, self).__init__(**kwargs)

    def _download(self, download_destination):
        """
        Method downloades file from config URL to given directory.

        Parameters
        ----------
        download_destination : str
            place where to download resource
        """
        SCPDownloader.download(
            uri=self.config[LargeResource.URI],
            path=download_destination,
            overwrite=False,
            **self._scp_config,
        )


def init_scp_large_resource_from_kwargs(resource, uri, archive, scp_host, user_dict):
    """
    Method initializes scp resource from resource informations and user
    credentials.

    Parameters
    ----------
    resource : str
        resource name, same as LargeResource.RESOURCE_NAME
    uri : str
        resource uri, same as LargeResource.URI
    archive : str
        archive type, see LargeResource.ARCHIVE
    scp_host : str
        remote host adress, see SCPLargeResource.SCP_HOST_KEY
    user_dict : dict(str, str)
        user dictionary that may contain scp_user that defines username,
        scp_private_key that defines path to private key, scp_pass_key that defines user
        password
    """

    if SCPLargeResource.SCP_USER_KEY not in user_dict:
        # if your username is same as the one on the server
        scp_user = getpass.getuser()
    else:
        scp_user = user_dict[SCPLargeResource.SCP_USER_KEY]

    scp_private_key = user_dict.get(SCPLargeResource.SCP_PRIVATE_KEY, None)
    scp_pass_key = user_dict.get(SCPLargeResource.SCP_PASS_KEY, None)

    config = {
        LargeResource.URI: uri,
        LargeResource.RESOURCE_NAME: resource,
        LargeResource.ARCHIVE: archive,
        SCPLargeResource.SCP_HOST_KEY: scp_host,
        SCPLargeResource.SCP_USER_KEY: scp_user,
        SCPLargeResource.SCP_PRIVATE_KEY: scp_private_key,
        SCPLargeResource.SCP_PASS_KEY: scp_pass_key,
    }
    SCPLargeResource(**config)
