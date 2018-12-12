"""Module contains storage utility methods."""
import os
import zipfile
import tarfile
from tqdm import tqdm


def copyfileobj_with_tqdm(finput, foutput, total_size, buffer_size=16*1024):
    """Function copies file like input finput to file like output foutput.
    Total size is used to display progress bar and buffer size to determine
    size of the buffer used for copying. The implementation is based on
    shutil.copyfileobj.

    Parameters
    ----------
    finput : file like object
        input object from which to copy the data
    foutput : file like object
        output object to which the data is copied
    total_size : int
        total input file size used for computing progress and displaying
        progress bar
    buffer_size : int
        constant used for determining maximal buffer size
    """
    with tqdm(total=total_size, unit='B', unit_scale=1) as progress:
        buffer = finput.read(buffer_size)
        while buffer:
            foutput.write(buffer)
            progress.update(len(buffer))
            buffer = finput.read(buffer_size)


def extract_zip_file(archive_file, destination_file):
    """Method extracts zip archive to destination.

    Parameters
    ----------
    archive_file : str
        path to the archive file that needs to be extracted
    destination_file : str
        path where file needs to be decompressed

    Raises
    ------
    ValueError
        if given archive file doesn't exists
    """
    if not os.path.exists(archive_file):
        raise ValueError("Given archive file doesn't exists. Given {}.".format(
            archive_file
        ))
    zip_ref = zipfile.ZipFile(file=archive_file, mode='r')
    zip_ref.extractall(path=destination_file)
    zip_ref.close()


def extract_tar_file(archive_file, destination_file):
    """Method extracts tar archive to destination, including those archives
    that are created using gzip, bz2 and lzma compression.

    Parameters
    ----------
    archive_file : str
        path to the archive file that needs to be extracted
    destination_file : str
        path where file needs to be decompressed

    Raises
    ------
    ValueError
        if given archive file doesn't exists
    """
    if not os.path.exists(archive_file):
        raise ValueError("Given archive file doesn't exists. Given {}.".format(
            archive_file
        ))
    tar_ref = tarfile.TarFile(name=archive_file, mode='r')
    tar_ref.extractall(path=destination_file)
    tar_ref.close()
