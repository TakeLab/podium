import tempfile
import shutil
import os

from takepod.dataload.imdb import Imdb


def write_to_file(path, text):
    with open(path, "w") as f:
        f.write(text)


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def test_load_dataset(tmpdir):
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    os.makedirs(os.path.join(base, "imdb", "aclImdb"))
    os.makedirs(os.path.join(base, "imdb", "aclImdb", "train", "neg"))
    os.makedirs(os.path.join(base, "imdb", "aclImdb", "train", "pos"))
    os.makedirs(os.path.join(base, "imdb", "aclImdb", "test", "neg"))
    os.makedirs(os.path.join(base, "imdb", "aclImdb", "test", "pos"))

    neg_tr = os.path.join(base, "imdb", "aclImdb", "train", "neg", "1.txt")
    neg_te = os.path.join(base, "imdb", "aclImdb", "test", "neg", "1.txt")
    pos_tr = os.path.join(base, "imdb", "aclImdb", "train", "pos", "1.txt")
    pos_te = os.path.join(base, "imdb", "aclImdb", "test", "pos", "1.txt")

    write_to_file(neg_tr, "neg_tr")
    write_to_file(neg_te, "neg_te")
    write_to_file(pos_tr, "pos_tr")
    write_to_file(pos_te, "pos_te")

    imdb = Imdb(path=base)
    X, y = imdb.load_data(train=True)

    assert X == ["pos_tr", "neg_tr"]
    assert y == ['Positive', 'Negative']

    shutil.rmtree(base)
    assert not os.path.exists(base)
