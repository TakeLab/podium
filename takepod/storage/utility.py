"""Module contains storage utility methods."""
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
