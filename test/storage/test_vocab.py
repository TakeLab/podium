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


def test_specials_get_pad_symbol():
    voc = vocab.Vocab(specials=(vocab.SpecialVocabSymbols.PAD,))
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data))
    assert voc.pad_symbol() == 0
    voc.finalize()
    assert voc.itos[0] == vocab.SpecialVocabSymbols.PAD


def test_max_size():
    voc = vocab.Vocab(max_size=2, specials=[])
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()
    assert len(voc) == 2


def test_max_size_with_specials():
    voc = vocab.Vocab(max_size=2, specials=[vocab.SpecialVocabSymbols.PAD,
                                            vocab.SpecialVocabSymbols.UNK])
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()
    assert len(voc) == 2


def test_size_after_final_with_specials():
    specials = [vocab.SpecialVocabSymbols.PAD, vocab.SpecialVocabSymbols.UNK]
    voc = vocab.Vocab(specials=specials)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()
    assert len(voc) == len(data) + len(specials)


def test_enum_special_vocab_symbols():
    assert vocab.SpecialVocabSymbols.PAD == "<pad>"
    assert vocab.SpecialVocabSymbols.UNK == "<unk>"


def test_get_stoi_for_unknown_word_default_unk():
    specials = [vocab.SpecialVocabSymbols.PAD, vocab.SpecialVocabSymbols.UNK]
    voc = vocab.Vocab(specials=specials)
    data = ["tree", "plant", "grass"]
    voc = (voc + set(data)) + {"plant"}
    voc.finalize()
    assert voc.stoi["unknown"] == 1


def test_add_word_after_finalization_error():
    voc = vocab.Vocab()
    voc.finalize()
    with pytest.raises(RuntimeError):
        voc = voc + {"word"}


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
    voc1 = (voc1 + set(data1)) + set(data3)
    voc2 = vocab.Vocab()
    voc2 += set(data2)

    voc = voc1 + voc2  # voc1 should be changed also
    assert voc == voc1
    for word in voc._freqs:
        assert voc._freqs[word] == expected_freq[word]


def test_add_list_word_to_vocab():
    voc = vocab.Vocab()
    voc += ["word", "word", "light", "heavy"]
    assert len(voc) == 3
    assert voc._freqs["word"] == 2


@pytest.mark.parametrize(
    "object_to_add",
    [
        1,
        1.566,
        "word"
    ]
)
def test_add_not_set_or_vocab_to_vocab_error(object_to_add):
    voc = vocab.Vocab()
    with pytest.raises(TypeError):
        voc += object_to_add


def test_finalize_finalized_vocab_error():
    voc = vocab.Vocab()
    voc.finalize()
    with pytest.raises(RuntimeError):
        voc.finalize()


def test_skip_stop_words():
    stop_words = ["the", "is", "a"]
    data = "the list is great".split(" ")
    voc = vocab.Vocab(stop_words=stop_words, specials=[])
    voc += data
    voc.finalize()
    assert len(voc) == 2


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


def test_vocab_fail_no_unk():
    voc = vocab.Vocab(specials=())
    voc += [1, 2, 3, 4, 5]
    voc.finalize()

    with pytest.raises(ValueError):
        voc.numericalize([1, 2, 3, 6])

def test_vocab_has_no_special():
    voc1 = vocab.Vocab(specials=None)
    assert not voc1.has_specials

    voc2 = vocab.Vocab(specials=())
    assert not voc2.has_specials

def test_vocab_has_specials():
    voc = vocab.Vocab()
    assert voc.has_specials
