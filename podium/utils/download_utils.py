import hashlib
import gzip
import zipfile
import logging
from typing import Optional
from dataclasses import dataclass

import requests
from tqdm.auto import tqdm


def _is_gzip(path: str) -> bool:
    # https://stackoverflow.com/a/60634210
    with gzip.open(path, "r") as fh:
        try:
            fh.read(1)
            return True
        except OSError:
            return False


def _is_xz(path: str) -> bool:
    # return True if the first six bytes
    # are equal to [0xFD, '7', 'z', 'X', 'Z', 0x00]
    with open(path, "rb") as f:
        try:
            header_magic_bytes = f.read(6)
        except OSError:
            return False
    return header_magic_bytes == b"\xfd7zXZ\x00"


def _is_rarfile(path: str) -> bool:
    # https://github.com/markokr/rarfile/blob/master/rarfile.py
    RAR_ID = b"Rar!\x1a\x07\x00"
    RAR5_ID = b"Rar!\x1a\x07\x01\x00"

    with open(path, "rb", 1024) as fd:
        buf = fd.read(len(RAR5_ID))
    return buf.startswith(RAR_ID) or buf.startswith(RAR5_ID)


# poredati funkcije
def get_checksum(path: str) -> str:
    # provjera a postoji file
    m = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


@dataclass
class DownloadConfig:
    filename: Optional[str] = None
    checksum: Optional[str] = None
    archive_type: str
    max_retries: int = 0


def download(
    url: str,
    destination: str,
    config: Optional[DownloadConfig] = None,
    **kwargs
) -> None:
    if config is None:
        config = DownloadConfig(**kwargs)
