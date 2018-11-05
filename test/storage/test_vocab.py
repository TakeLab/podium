import pytest
from takepod.storage import vocab


def test_default_vocab_add_set_words():
    voc = vocab.Vocab()
    voc = voc + {"tree", "plant", "grass"}

    assert len(voc._freqs) == 3
    assert voc._freqs["tree"] == 1

    voc = voc + {"tree"}
    assert len(voc._freqs) == 3
    assert voc._freqs["tree"] == 2


def test_default_vocab_iadd_set_words():
    voc = vocab.Vocab()
    voc += {"tree", "plant", "grass"}
    voc += {"tree"}

    assert len(voc._freqs) == 3
    assert voc._freqs["tree"] == 2
    assert voc._freqs["plant"] == 1


def test_get_frequency():
    voc = vocab.Vocab(keep_freqs=True)
    voc = (voc + {"tree", "plant", "grass"})+{"plant"}
    assert voc.get_freqs()["tree"] == 1
    assert voc.get_freqs()["plant"] == 2


def test_get_frequency_not_kept():
    voc = vocab.Vocab()
    with pytest.raises(RuntimeError):
        voc.get_freqs()


def test_numericalize_not_finalized():
    voc = vocab.Vocab(keep_freqs=True)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data))+{"plant"}

    with pytest.raises(RuntimeError):
        voc.numericalize(data)


def test_vocab_len_not_finalized():
    voc = vocab.Vocab(keep_freqs=True)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data))+{"plant"}
    assert len(voc) == len(data)

def test_empty_specials_len():
    voc = vocab.Vocab(specials=[])
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data))
    voc.finalize()

    assert len(voc) == 3
    assert len(voc.stoi) == 3
    assert len(voc.itos) == 3

def test_empty_specials_get_pad_symbol():
    voc = vocab.Vocab(specials=[])
    voc.finalize()
    with pytest.raises(ValueError):
        voc.pad_symbol()

def test_empty_specials_stoi():
    voc = vocab.Vocab(specials=[])
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data))
    voc.finalize()
    with pytest.raises(ValueError):
        voc.stoi["apple"]

def test_max_size():
    voc = vocab.Vocab(max_size=2, specials=[])
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data))+{"plant"}