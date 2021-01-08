"""
Package contains modules for storing and loading datasets and vectors.
"""

from .resources.downloader import (
    BaseDownloader,
    HttpDownloader,
    SCPDownloader,
    SimpleHttpDownloader,
)
from .resources.large_resource import LargeResource, SCPLargeResource
