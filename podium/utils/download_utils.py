import gzip
import hashlib
import logging
import lzma
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, ContextManager, Dict, Optional, Union
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import requests
from tqdm.auto import tqdm


__all__ = [
    "download",
    "DownloadConfig",
    "get_checksum",
]

try:
    import rarfile
except ImportError:
    _is_rarfile_available = False
else:
    _is_rarfile_available = True


class UnsupportedArchiveType(Exception):
    pass


def is_gzip(path: str) -> bool:
    # https://stackoverflow.com/a/60634210
    with gzip.open(path, "r") as fh:
        try:
            fh.read(1)
            return True
        except OSError:
            return False


def is_xz(path: str) -> bool:
    with open(path, "rb") as f:
        try:
            header_magic_bytes = f.read(6)
        except OSError:
            return False
    return header_magic_bytes == b"\xfd7zXZ\x00"


def is_rarfile(path: str) -> bool:
    # https://github.com/markokr/rarfile/blob/master/rarfile.py
    RAR_ID = b"Rar!\x1a\x07\x00"
    RAR5_ID = b"Rar!\x1a\x07\x01\x00"

    with open(path, "rb", 1024) as fd:
        buf = fd.read(len(RAR5_ID))
    return buf.startswith(RAR_ID) or buf.startswith(RAR5_ID)


def is_url(path: str) -> bool:
    return urlparse(path).scheme in ("ftp", "gs", "http", "https", "s3")


def get_checksum(path: str) -> str:
    m = hashlib.sha256()

    def get_file_checksum(filename):
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                m.update(chunk)

    if os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for filename in sorted(filenames):
                get_file_checksum(os.path.join(dirpath, filename))
    else:
        get_file_checksum(path)
    return m.hexdigest()


def _request_with_retry(
    verb: str,
    url: str,
    max_retries: int = 0,
    base_wait_time: float = 0.5,
    max_wait_time: float = 2,
    **params,
) -> requests.Response:
    tries, success = 0, False
    while not success:
        tries += 1
        try:
            response = requests.request(verb, url, **params)
            success = True
        except requests.exceptions.ConnectTimeout as err:
            if tries > max_retries:
                raise ConnectionError("Unable to download resource using HTTP") from err
            else:
                logging.info(
                    f"{verb} request to {url} timed out, retrying... [{tries/max_retries}]"
                )
                sleep_time = max(
                    max_wait_time, base_wait_time * 2 ** (tries - 1)
                )  # exponential backoff
                time.sleep(sleep_time)
    return response


def ftp_head(url: str, timeout: float = 10.0) -> bool:
    try:
        with closing(urllib.request.urlopen(url, timeout=timeout)) as r:
            r.read(1)
    except Exception:
        return False
    return True


def ftp_get(url: str, temp_file: BinaryIO, timeout: float = 10.0) -> None:
    try:
        with closing(urllib.request.urlopen(url, timeout=timeout)) as r:
            shutil.copyfileobj(r, temp_file)
    except urllib.error.URLError as err:
        raise ConnectionError("Unable to download resource using FTP") from err


def http_head(
    url: str,
    cookies: Optional[Union[Dict[str, str], requests.cookies.RequestsCookieJar]] = None,
    timeout: float = 10.0,
    max_retries: int = 0,
) -> requests.Response:
    response = _request_with_retry(
        verb="HEAD",
        url=url,
        cookies=cookies,
        allow_redirects=True,
        timeout=timeout,
        max_retries=max_retries,
    )
    return response


def http_get(url: str, temp_file: BinaryIO, cookies=None, max_retries: int = 0) -> None:
    response = _request_with_retry(
        verb="GET",
        url=url,
        stream=True,
        cookies=cookies,
        max_retries=max_retries,
    )
    content_length = response.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    disable = logging.root.level > logging.WARNING
    with tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        desc="Downloading resource",
        disable=disable,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress_bar.update(len(chunk))
                temp_file.write(chunk)


def download_from_url(
    url: str,
    temp_file: BinaryIO,
    max_retries: int,
) -> None:
    cookies = None

    if url.startswith("ftp://"):
        logging.info("Initiating FTP connection")
        connected = ftp_head(url)
        if connected:
            ftp_get(url, temp_file)
        else:
            raise ConnectionError(f"Unable to connect to {url}")
    else:
        logging.info("Initiating HTTP connection")
        response = http_head(url, max_retries=max_retries)

        if response.status_code == 200:
            for k, v in response.cookies.items():
                if k.startswith("download_warning") and "drive.google.com" in url:
                    url += "&confirm=" + v
                    cookies = response.cookies
            connected = True
        elif (
            (response.status_code == 400 and "firebasestorage.googleapis.com" in url)
            or (response.status_code == 405 and "drive.google.com" in url)
            or (
                response.status_code == 403
                and re.match(
                    r"^https?://github.com/.*?/.*?/releases/download/.*?/.*?$", url
                )
            )
        ):
            connected = True

        if connected:
            http_get(
                url,
                temp_file,
                cookies=cookies,
                max_retries=max_retries,
            )
        else:
            raise ConnectionError(f"Unable to connect to {url}")


def extract(source: str, destination: str) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    os.makedirs(destination, exist_ok=True)
    if tarfile.is_tarfile(source):
        with closing(tarfile.open(source, "r")) as tar_file:
            tar_file.extractall(destination)
    elif is_gzip(source):
        os.rmdir(destination)
        with gzip.open(source, "rb") as gzip_file:
            with open(destination, "wb") as extracted_file:
                shutil.copyfileobj(gzip_file, extracted_file)
    elif is_zipfile(source):
        with ZipFile(source, "r") as zip_file:
            zip_file.extractall(destination)
    elif is_xz(source):
        os.rmdir(destination)
        with lzma.open(source, "rb") as xz_file:
            with open(destination, "wb") as extracted_file:
                shutil.copyfileobj(xz_file, extracted_file)
    elif is_rarfile(source):
        if _is_rarfile_available:
            with rarfile.RarFile(source, "r") as rar_file:
                rar_file.extractall(destination)
        else:
            raise ImportError(
                "RAR file extraction requires having the rarfile package installed"
            )
    else:
        raise UnsupportedArchiveType(f"Could not identify archive type of {source}")


@dataclass
class DownloadConfig:
    extract: bool = False
    extract_function: Optional[Callable[[str, str], None]] = None
    checksum: Optional[str] = None
    max_retries: int = 0


def get_url_manager(url: str, max_retries: int = 0) -> ContextManager[str]:
    @contextmanager
    def url_resource():
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                download_from_url(url, temp_file, max_retries=max_retries)
            yield temp_file.name
        finally:
            os.remove(temp_file.name)
    return url_resource


def get_local_path_manager(filename: str) -> ContextManager[str]:
    @contextmanager
    def local_path_resource():
        try:
            yield filename
        finally:
            pass

    return local_path_resource


def download(
    url_or_filename: str,
    destination: str,
    config: Optional[DownloadConfig] = None,
    **kwargs,
) -> None:
    if config is None:
        config = DownloadConfig(**kwargs)

    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    if is_url(url_or_filename):
        resource_manager = get_url_manager(url_or_filename, config.max_retries)
    elif os.path.exists(url_or_filename):
        resource_manager = get_local_path_manager(url_or_filename)
    else:
        raise ValueError(f"{url_or_filename} is neither an URL nor a local path")

    with resource_manager() as source:
        if config.checksum is not None:
            logging.info(f"Verifying integrity of {source}")
            if get_checksum(source) != config.checksum:
                raise ValueError("Checksums don't match")

        if config.extract:
            logging.info("Extracting...")
            try:
                extract(source, destination)
            except UnsupportedArchiveType:
                if config.extract_function is not None:
                    config.extract_function(source, destination)
                else:
                    raise
        else:
            logging.info(f"Moving {source} to {destination}")
            shutil.rmtree(destination, ignore_errors=True)
            os.makedirs(destination, exist_ok=True)
            os.rmdir(destination)
            shutil.move(source, destination)
