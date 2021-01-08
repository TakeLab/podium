import os

import dill
import numpy as np
import pytest

from podium import vocab


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
    voc = (voc + {"tree", "plant", "grass"}) + {"plant"}
    assert voc.get_freqs()["tree"] == 1
    assert voc.get_freqs()["plant"] == 2


def test_get_frequency_not_kept_not_finalized():
    voc = vocab.Vocab(keep_freqs=False)
    voc = (voc + {"tree", "plant", "grass"}) + {"plant"}
    assert voc.get_freqs()["tree"] == 1
    assert voc.get_freqs()["plant"] == 2


def test_vocab_iterable_not_finalized():
    voc = vocab.Vocab(keep_freqs=False)
    data = {"tree", "plant", "grass"}
    voc += data
    elements = []
    for i in voc:
        elements.append(i)
    assert all([e in data for e in elements])


def test_vocab_iterable_finalized():
    voc = vocab.Vocab(keep_freqs=False, specials=())
    data = {"tree", "plant", "grass"}
    voc += data
    voc.finalize()
    elements = []
    for i in voc:
        elements.append(i)
    assert all([e in data for e in elements])


def test_get_frequency_not_kept_finalized():
    voc = vocab.Vocab(keep_freqs=False)
    voc.finalize()
    with pytest.raises(RuntimeError):
        voc.get_freqs()


def test_numericalize_not_finalized():
    voc = vocab.Vocab(keep_freqs=True)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}

    with pytest.raises(RuntimeError):
        voc.numericalize(data)


def test_vocab_len_not_finalized():
    voc = vocab.Vocab(keep_freqs=True)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    assert len(voc) == len(data)


def test_empty_specials_len():
    voc = vocab.Vocab(specials=[])
    data = ["tree", "plant", "grass"]
    voc = voc + set(data)
    voc.finalize()

    assert len(voc) == 3
    assert len(voc.stoi) == 3
    assert len(voc.itos) == 3


def test_empty_specials_get_pad_symbol():
    voc = vocab.Vocab(specials=[])
    voc.finalize()
    with pytest.raises(ValueError):
        voc.padding_index()


def test_no_unk_filters_unknown_tokens():
    voc = vocab.Vocab(specials=[])
    data = ["tree", "plant", "grass"]
    voc = voc + set(data)
    voc.finalize()

    # Tree is in vocab
    assert len(voc.numericalize("tree")) == 1
    # Apple isn't in vocab
    assert len(voc.numericalize("apple")) == 0
    # Try with list argument
    assert len(voc.numericalize(["tree", "apple"])) == 1


def test_specials_uniqueness():
    with pytest.raises(ValueError):
        vocab.Vocab(specials=[vocab.UNK(), vocab.UNK()])

    with pytest.raises(ValueError):
        vocab.Vocab(specials=[vocab.UNK(), vocab.UNK("<my_unknown>")])

    with pytest.raises(ValueError):
        vocab.Vocab(specials=[vocab.PAD(), vocab.PAD()])

    with pytest.raises(ValueError):
        vocab.Vocab(specials=[vocab.PAD(), vocab.PAD("<my_pad>")])

    with pytest.raises(ValueError):
        vocab.Vocab(specials=[vocab.BOS(), vocab.BOS()])

    with pytest.raises(ValueError):
        vocab.Vocab(specials=[vocab.BOS(), vocab.BOS("<my_bos>")])

    with pytest.raises(ValueError):
        vocab.Vocab(specials=[vocab.EOS(), vocab.EOS()])

    with pytest.raises(ValueError):
        vocab.Vocab(specials=[vocab.EOS(), vocab.EOS("<my_eos>")])


def test_specials_get_pad_symbol():
    voc = vocab.Vocab(specials=(vocab.PAD(),))
    data = ["tree", "plant", "grass"]
    voc = voc + set(data)
    assert voc.padding_index() == 0
    voc.finalize()
    assert voc.itos[0] == vocab.PAD()


def test_max_size():
    voc = vocab.Vocab(max_size=2, specials=[])
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()
    assert len(voc) == 2


def test_max_size_with_specials():
    voc = vocab.Vocab(
        max_size=2,
        specials=[vocab.PAD(), vocab.UNK()],
    )
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()
    assert len(voc) == 2


def test_size_after_final_with_specials():
    specials = [vocab.PAD(), vocab.UNK()]
    voc = vocab.Vocab(specials=specials)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()
    assert len(voc) == len(data) + len(specials)


def test_special_vocab_symbols():
    assert str(vocab.PAD()) == "<PAD>"
    assert str(vocab.UNK()) == "<UNK>"

    assert str(vocab.PAD("<my_pad>")) == "<my_pad>"
    assert str(vocab.UNK("<my_unk>")) == "<my_unk>"

    # These hold due to overloaded hash/eq
    assert vocab.PAD("<my_pad>") == vocab.PAD()
    assert vocab.UNK("<my_unk>") == vocab.UNK()


def test_get_stoi_for_unknown_word_default_unk():
    specials = [vocab.PAD(), vocab.UNK()]
    voc = vocab.Vocab(specials=specials)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()
    assert voc.numericalize("unknown") == 1


def test_iadd_word_after_finalization_error():
    voc = vocab.Vocab()
    voc.finalize()
    with pytest.raises(RuntimeError):
        voc += {"word"}


def test_add_vocab_to_vocab():
    data1 = ["w1", "w2", "w3"]
    data2 = ["a1", "a2"]
    data3 = ["w1", "a2"]
    expected_freq = {"w1": 2, "w2": 1, "w3": 1, "a1": 1, "a2": 2}

    voc1 = vocab.Vocab()
    voc1 += data1
    voc1 += data3

    voc2 = vocab.Vocab()
    voc2 += data2

    voc = voc1 + voc2
    assert not voc.finalized
    for word in voc._freqs:
        assert voc._freqs[word] == expected_freq[word]

    voc3 = vocab.Vocab(specials=vocab.UNK())
    voc3 += data1
    voc3 += data3
    voc3.finalize()

    voc4 = vocab.Vocab(specials=vocab.PAD())
    voc4 += data2
    voc4.finalize()

    voc = voc3 + voc4
    assert set(voc.specials) == {
        vocab.PAD(),
        vocab.UNK(),
    }
    assert voc.finalized
    assert len(voc.itos) == 7


def test_iadd_vocab_to_vocab():
    data1 = ["w1", "w2", "w3"]
    data2 = ["a1", "a2", "w1"]
    expected_freqs = {"w1": 2, "w2": 1, "w3": 1, "a1": 1, "a2": 1}

    voc1 = vocab.Vocab(specials=vocab.PAD())
    voc1 += data1

    voc2 = vocab.Vocab(specials=vocab.UNK())
    voc2 += data2

    voc1 += voc2

    assert voc1.get_freqs() == expected_freqs
    assert all(spec in voc1.specials for spec in (vocab.PAD(), vocab.UNK()))


def test_add_vocab_to_vocab_error():
    voc1 = vocab.Vocab()
    voc2 = vocab.Vocab()

    voc1 += ["one", "two"]
    voc2 += ["three", "four"]

    voc1.finalize()

    # This should cause an error because the finalization state of the vocabs differs
    with pytest.raises(RuntimeError):
        voc1 + voc2

    with pytest.raises(RuntimeError):
        voc2 + voc1


def test_add_list_word_to_vocab():
    voc = vocab.Vocab()
    voc += ["word", "word", "light", "heavy"]
    assert len(voc) == 3
    assert voc._freqs["word"] == 2


def test_add_non_iterable_object_to_vocab():
    class NonIterableClass:
        pass

    voc = vocab.Vocab()

    with pytest.raises(TypeError):
        voc + NonIterableClass()


def test_iadd_non_iterable_object_to_vocab():
    class NonIterableClass:
        pass

    voc = vocab.Vocab()

    with pytest.raises(TypeError):
        voc += NonIterableClass()


@pytest.mark.parametrize("object_to_add", [1, 1.566, "word"])
def test_add_not_set_or_vocab_to_vocab_error(object_to_add):
    voc = vocab.Vocab()
    with pytest.raises(TypeError):
        voc += object_to_add


def test_numericalize():
    voc = vocab.Vocab(specials=[])
    voc += ["word", "word", "aaa"]
    voc.finalize()
    data = ["word", "aaa", "word"]
    word_num = voc.numericalize(data)
    for i in range(len(data)):
        assert voc.stoi[data[i]] == word_num[i]


def test_equals_two_vocabs():
    data = ["a", "b"]
    voc1 = vocab.Vocab()
    voc1 += data
    voc2 = vocab.Vocab()
    voc2 += data
    assert voc1 == voc2
    voc1.finalize()
    voc2.finalize()
    assert voc1 == voc2


def test_vocab_pickle(tmpdir):
    data = ["a", "b"]
    voc = vocab.Vocab()
    voc += data

    vocab_file = os.path.join(tmpdir, "vocab.pkl")

    with open(vocab_file, "wb") as fdata:
        dill.dump(voc, fdata)

    with open(vocab_file, "rb") as fdata:
        loaded_voc = dill.load(fdata)

        assert voc == loaded_voc


def test_finalized_vocab_pickle(tmpdir):
    data = ["a", "b"]
    voc = vocab.Vocab()
    voc += data

    vocab_file = os.path.join(tmpdir, "vocab.pkl")

    with open(vocab_file, "wb") as fdata:
        dill.dump(voc, fdata)

    with open(vocab_file, "rb") as fdata:
        loaded_voc = dill.load(fdata)

        assert voc == loaded_voc


def test_equals_two_vocabs_different_finalization():
    data = ["a", "b"]
    voc1 = vocab.Vocab()
    voc1 += data
    voc1.finalize()
    voc2 = vocab.Vocab()
    voc2 += data
    assert voc1 != voc2


def test_equals_two_vocabs_different_freq():
    data = ["a", "b"]
    voc1 = vocab.Vocab()
    voc1 += data
    voc1.finalize()
    voc2 = vocab.Vocab()
    voc2 += data
    voc2 += ["a"]
    assert voc1 != voc2


# This won't fail anymore, should change to
# test_vocab_filer_unk
def test_vocab_fail_no_unk():
    voc = vocab.Vocab(specials=())
    voc += [1, 2, 3, 4, 5]
    voc.finalize()

    assert np.array_equal(voc.numericalize([1, 2, 3, 6]), np.array([0, 1, 2]))


def test_vocab_has_no_specials():
    voc1 = vocab.Vocab(specials=None)
    assert not voc1.has_specials

    voc2 = vocab.Vocab(specials=())
    assert not voc2.has_specials


def test_vocab_has_specials():
    voc = vocab.Vocab()
    assert voc.has_specials

    voc2 = vocab.Vocab(specials=vocab.UNK())
    assert voc2._has_specials
    assert voc2.specials == (vocab.UNK(),)


def test_reverse_numericalize():
    words = ["first", "second", "third"]

    voc = vocab.Vocab()
    voc += words
    voc.finalize()

    assert words == voc.reverse_numericalize(voc.numericalize(words))


def test_reverse_numericalize_not_finalized():
    words = ["first", "second", "third"]

    voc = vocab.Vocab()
    voc += words

    with pytest.raises(RuntimeError):
        voc.reverse_numericalize(voc.numericalize(words))


def test_vocab_static_constructors():
    specials = [vocab.PAD(), vocab.UNK()]
    voc = vocab.Vocab(specials=specials)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()

    itos2voc = vocab.Vocab.from_itos(voc.itos)
    # Only the frequencies will be different because
    # we don't transfer this information, so the full
    # vocab1 == vocab2 will fail. Perhaps split equality
    # checks for vocab on before/after finalization?

    assert itos2voc.itos == voc.itos
    assert itos2voc.stoi == voc.stoi
    assert itos2voc.specials == voc.specials

    stoi2voc = vocab.Vocab.from_stoi(voc.stoi)
    assert stoi2voc.itos == voc.itos
    assert stoi2voc.stoi == voc.stoi
    assert stoi2voc.specials == voc.specials
