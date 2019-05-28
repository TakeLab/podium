import os
import dill
import numpy as np
import pytest
from mock import patch

from takepod.storage import Field, TokenizedField, MultilabelField, \
    Vocab, SpecialVocabSymbols

ONE_TO_FIVE = [1, 2, 3, 4, 5]

CUSTOM_PAD = 43

PAD_NUM = 42


class MockToken:
    def __init__(self, txt):
        self.text = txt


class MockSpacy:
    def load(self, x, **kwargs):
        class MockTokenizer:
            def tokenizer(self, string):
                return [MockToken(tok) for tok in string.split()]

        return MockTokenizer()


class MockVocab:
    def __init__(self):
        self.values = []
        self.finalized = False
        self.numericalized = False

    def pad_symbol(self):
        return PAD_NUM

    def __add__(self, values):
        if type(values) == type(self):
            pass
        else:
            self.values.extend(values)

        return self

    def __iadd__(self, other):
        return self.__add__(other)

    def finalize(self):
        if self.finalized:
            raise Exception
        else:
            self.finalized = True

    def numericalize(self, data):
        self.numericalized = True


@pytest.fixture()
def vocab():
    return MockVocab()


def test_field_store_raw_sequential_exception():
    with pytest.raises(ValueError):
        Field(name="F", store_as_raw=False, tokenize=False)


def test_field_preprocess_eager(vocab):
    f = Field(name="F", vocab=vocab, eager=True)
    f.preprocess("some text")

    # vocab was updated
    assert len(vocab.values) > 0


@pytest.mark.parametrize(
    "value, store_raw, sequential, expected_raw_value, "
    "expected_tokenized_value",
    [
        ("some text", True, True, "some text", ["some", "text"]),
        ("some text", True, False, "some text", None),
        ("some text", False, True, None, ["some", "text"]),
    ]
)
def test_field_preprocess_raw_sequential(value, store_raw, sequential,
                                         expected_raw_value,
                                         expected_tokenized_value):
    f = Field(name="F", store_as_raw=store_raw, tokenize=sequential)

    received_raw_value, received_tokenized_value = f.preprocess(value)

    assert received_raw_value == expected_raw_value
    assert received_tokenized_value == expected_tokenized_value


@pytest.mark.parametrize(
    "value, store_raw, sequential, expected_raw_value, "
    "expected_tokenized_value",
    [
        ("some text", True, True, "some text", ["some", "text"]),
        ("some text", True, False, "some text", None),
        ("some text", False, True, None, ["some", "text"]),
    ]
)
def test_field_pickle_tokenized(value, store_raw, sequential,
                                expected_raw_value,
                                expected_tokenized_value, tmpdir):
    fld = Field(name="F", store_as_raw=store_raw, tokenize=sequential)

    received_raw_value, received_tokenized_value = fld.preprocess(value)

    assert received_raw_value == expected_raw_value
    assert received_tokenized_value == expected_tokenized_value

    field_file = os.path.join(tmpdir, "field.pkl")

    with open(field_file, "wb") as fdata:
        dill.dump(fld, fdata)

    with open(field_file, "rb") as fdata:
        loaded_fld = dill.load(fdata)
        raw_value, tokenized_value = loaded_fld.preprocess(value)

        assert raw_value == expected_raw_value
        assert tokenized_value == expected_tokenized_value
        assert loaded_fld.name == "F"
        assert loaded_fld.store_as_raw == store_raw
        assert loaded_fld.sequential == sequential


@pytest.mark.parametrize(
    "vocab, expected_value",
    [
        (vocab(), True),
        (None, False)
    ]
)
def test_field_use_vocab(vocab, expected_value):
    f = Field(name="F", vocab=vocab)

    assert f.use_vocab == expected_value


@pytest.mark.parametrize(
    "use_vocab, sequential, expected_vocab_values",
    [
        (False, False, []),
        (False, True, []),
        (True, False, ["some text"]),
        (True, True, ["some", "text"]),
    ]
)
def test_field_update_vocab(use_vocab, sequential, expected_vocab_values,
                            vocab):
    f = Field(name="F", vocab=vocab if use_vocab else None,
              tokenize=sequential)

    raw_value = "some text"
    tokenized_value = ["some", "text"]

    f.update_vocab(raw_value, tokenized_value)

    assert vocab.values == expected_vocab_values


def test_field_finalize(vocab):
    f = Field(name="F", vocab=vocab)

    assert not vocab.finalized
    f.finalize()
    assert vocab.finalized
    with pytest.raises(Exception):
        f.finalize()


@pytest.mark.parametrize(
    "use_vocab, expected_numericalized, custom_numericalize",
    [
        (False, False, float),
        (True, True, None),
    ]
)
def test_field_numericalize_vocab(use_vocab, expected_numericalized, vocab,
                                  custom_numericalize):
    f = Field(name="F", vocab=vocab if use_vocab else None, tokenize=False,
              custom_numericalize=custom_numericalize)
    f.numericalize(("4.32", None))

    assert vocab.numericalized == expected_numericalized


def test_field_custom_numericalize_with_vocab(vocab):
    f = Field(name="F", vocab=vocab, tokenize=False,
              custom_numericalize=float)
    numericalized = f.numericalize(("4.32", None))

    assert not vocab.numericalized
    assert abs(numericalized - 4.32) < 1e-6


@pytest.mark.parametrize(
    "row, length, expected_row, pad_left, truncate_left",
    [
        (ONE_TO_FIVE, 3, [1, 2, 3], False, False),
        (ONE_TO_FIVE, 3, [3, 4, 5], False, True),
        (ONE_TO_FIVE, 7, [1, 2, 3, 4, 5, PAD_NUM, PAD_NUM], False, False),
        (ONE_TO_FIVE, 7, [PAD_NUM, PAD_NUM, 1, 2, 3, 4, 5], True, False),
        (ONE_TO_FIVE, 5, ONE_TO_FIVE, False, False),
        (ONE_TO_FIVE, 5, ONE_TO_FIVE, True, True),
        (ONE_TO_FIVE, 5, ONE_TO_FIVE, True, False),
        (ONE_TO_FIVE, 5, ONE_TO_FIVE, False, True),
        (ONE_TO_FIVE, 0, [], False, False),
        (ONE_TO_FIVE, 0, [], False, True)
    ]
)
def test_field_pad_to_length(row, length, expected_row, vocab, pad_left,
                             truncate_left):
    f = Field(name="F", vocab=vocab)

    received_row = f.pad_to_length(np.array(row), length, pad_left=pad_left,
                                   truncate_left=truncate_left)

    assert received_row.tolist() == expected_row


@pytest.mark.parametrize(
    "row, length, expected_row",
    [
        (ONE_TO_FIVE, 3, [1, 2, 3]),
        (ONE_TO_FIVE, 7, [1, 2, 3, 4, 5, CUSTOM_PAD, CUSTOM_PAD]),
        (ONE_TO_FIVE, 5, ONE_TO_FIVE),
        (ONE_TO_FIVE, 0, [])
    ]
)
def test_field_pad_to_length_custom_pad(row, length, expected_row):
    f = Field(name="F", vocab=None)

    row_arr = np.array(row)
    received_row = f.pad_to_length(row_arr, length,
                                   custom_pad_symbol=CUSTOM_PAD)

    assert received_row.tolist() == expected_row


def test_field_pad_to_length_exception():
    # set vocab to be None
    f = Field(name="F", vocab=None)

    row_arr = np.array(ONE_TO_FIVE)
    length = 7

    custom_pad_symbol = None
    with pytest.raises(ValueError):
        f.pad_to_length(row_arr, length, custom_pad_symbol=custom_pad_symbol)


def test_field_get_tokenizer_callable(vocab):
    def my_tokenizer(string):
        return [string[0], string[1:]]

    f = Field(name="F", vocab=vocab, tokenizer=my_tokenizer, tokenize=True,
              store_as_raw=False)

    assert f.preprocess("asd dsa") == (None, ["a", "sd dsa"])


def test_field_get_tokenizer_spacy_exception(vocab):
    class MockSpacy:
        def load(self, x, **kwargs):
            raise OSError

    patch.dict("sys.modules", spacy=MockSpacy()).start()

    with pytest.raises(OSError):
        Field(name="F", vocab=vocab, tokenizer="spacy", tokenize=True)


def test_field_get_tokenizer_default(vocab):
    f = Field(name="F", vocab=vocab, tokenize=True, store_as_raw=False)

    assert f.preprocess("asd dsa") == (None, ["asd", "dsa"])


def test_field_get_tokenizer_exception(vocab):
    with pytest.raises(ValueError):
        Field(name="F", vocab=vocab, tokenizer="NOT_tokenizer",
              tokenize=True, store_as_raw=False)


def test_field_get_tokenizer_spacy_ok(vocab):
    patch.dict("sys.modules", spacy=MockSpacy()).start()
    f = Field(name="F", vocab=vocab, tokenizer="spacy", tokenize=True,
              store_as_raw=False)
    assert f.preprocess("bla blu") == (None, ["bla", "blu"])


def test_field_pickle_spacy_tokenizer(vocab, tmpdir):
    patch.dict("sys.modules", spacy=MockSpacy()).start()
    fld = Field(name="F", vocab=vocab, tokenizer="spacy", tokenize=True,
                store_as_raw=False)
    assert fld.preprocess("bla blu") == (None, ["bla", "blu"])

    field_file = os.path.join(tmpdir, "field.pkl")

    with open(field_file, "wb") as fdata:
        dill.dump(fld, fdata)

    with open(field_file, "rb") as fdata:
        loaded_fld = dill.load(fdata)

        assert loaded_fld._tokenizer_arg == "spacy"
        assert loaded_fld.preprocess("bla blu") == (None, ["bla", "blu"])


def test_field_pretokenize_hooks():
    f = Field(name="F", tokenize=True)

    f.add_pretokenize_hook(str.lower)
    f.add_pretokenize_hook(lambda x: x.replace("bla", "blu"))
    f.add_pretokenize_hook(lambda x: x.replace(";", " "))
    f.add_pretokenize_hook(lambda x: x.replace(",", " "))

    raw_str = "asd;123,BLA"

    received = f.preprocess(raw_str)
    expected = ("asd 123 blu", ["asd", "123", "blu"])

    assert received == expected


def test_field_pretokenize_hooks_detach():
    f = Field(name="F", tokenize=True)

    f.add_pretokenize_hook(str.lower)
    f.add_pretokenize_hook(lambda x: x.replace(";", " "))
    f.add_pretokenize_hook(lambda x: x.replace(",", " "))

    # detaching
    f.remove_pretokenize_hooks()

    raw_str = "asd;123,BLA"

    received = f.preprocess(raw_str)

    expected = (raw_str, [raw_str])

    assert received == expected


def test_field_posttokenize_hooks():
    f = Field(name="F", tokenize=True)

    def remove_tags_hook(raw, tokenized):
        raw = raw.replace("<tag>", "")
        tokenized = map(lambda x: x.replace("<tag>", ""), tokenized)

        return raw, tokenized

    def to_upper_hook(raw, tokenized):
        raw = raw.upper()
        tokenized = map(str.upper, tokenized)

        return raw, tokenized

    f.add_posttokenize_hook(remove_tags_hook)
    f.add_posttokenize_hook(to_upper_hook)

    received = f.preprocess("asd 123<tag> B<tag>LA")
    expected = ("ASD 123 BLA", ["ASD", "123", "BLA"])

    assert received == expected


def test_field_posttokenize_hooks_detach():
    f = Field(name="F", tokenize=True, custom_numericalize=float)

    def remove_tags_hook(raw, tokenized):
        raw = raw.replace("<tag>", "")
        tokenized = map(lambda x: x.replace("<tag>", ""), tokenized)

        return raw, tokenized

    def to_upper_hook(raw, tokenized):
        raw = raw.upper()
        tokenized = map(str.upper, tokenized)

        return raw, tokenized

    f.add_posttokenize_hook(remove_tags_hook)
    f.add_posttokenize_hook(to_upper_hook)

    # detaching the hooks
    f.remove_posttokenize_hooks()

    received = f.preprocess("asd 123<tag> B<tag>LA")
    expected = ("asd 123<tag> B<tag>LA", ["asd", "123<tag>", "B<tag>LA"])

    assert received == expected


def test_field_repeated_hooks():
    def replace_tag_hook(raw, tokenized):
        replaced_tags = map(lambda s: s.replace("<tag>", "ABC"), tokenized)

        return raw, replaced_tags

    def to_lower_hook(raw, tokenized):
        # keep track of the function call count
        to_lower_hook.call_count += 1

        tokenized = map(str.lower, tokenized)

        return raw, tokenized

    to_lower_hook.call_count = 0

    f = Field(name="F", tokenize=True, custom_numericalize=float)

    # TAG -> tag
    f.add_posttokenize_hook(to_lower_hook)

    # <tag> -> ABC
    f.add_posttokenize_hook(replace_tag_hook)

    # ABC -> abc
    f.add_posttokenize_hook(to_lower_hook)

    received = f.preprocess("BLA <TAG> bla")

    expected = ("BLA <TAG> bla", ["bla", "abc", "bla"])

    assert received == expected

    # check that the hook that was added twice was also called twice
    assert to_lower_hook.call_count == 2


def test_field_is_target():
    f1 = Field(name="text", is_target=False)
    f2 = Field(name="label", is_target=True)
    f3 = Field(name="bla")

    assert not f1.is_target
    assert f2.is_target
    assert not f3.is_target


def test_tokenized_field_numericalization():
    vocab = Vocab()
    pretokenized_input1 = [
        "word",
        "words",
        "uttering"
    ]
    pretokenized_input2 = [
        "word",
        "words"
    ]
    pretokenized_input3 = [
        "word"
    ]

    pretokenized_input4 = [
        "word",
        "uttering"
    ]

    tokenized_field = TokenizedField("test_field", vocab=vocab)

    data1 = tokenized_field.preprocess(pretokenized_input1)
    data2 = tokenized_field.preprocess(pretokenized_input2)
    data3 = tokenized_field.preprocess(pretokenized_input3)
    data4 = tokenized_field.preprocess(pretokenized_input4)

    tokenized_field.finalize()

    expected_numericalization_1 = np.array([2, 3, 4])
    _, tok1 = data1
    assert np.all(vocab.numericalize(tok1) == expected_numericalization_1)
    assert np.all(tokenized_field.numericalize(data1) == expected_numericalization_1)

    expected_numericalization_2 = np.array([2, 3])
    _, tok2 = data2
    assert np.all(vocab.numericalize(tok2) == expected_numericalization_2)
    assert np.all(tokenized_field.numericalize(data2) == expected_numericalization_2)

    expected_numericalization_3 = np.array([2])
    _, tok3 = data3
    assert np.all(vocab.numericalize(tok3) == expected_numericalization_3)
    assert np.all(tokenized_field.numericalize(data3) == expected_numericalization_3)

    expected_numericalization_4 = np.array([2, 4])
    _, tok4 = data4
    assert np.all(vocab.numericalize(tok4) == expected_numericalization_4)
    assert np.all(tokenized_field.numericalize(data4) == expected_numericalization_4)


def test_tokenized_field_custom_numericalization():
    tfield = TokenizedField("bla",
                            custom_numericalize=lambda x: x)

    data1 = tfield.preprocess([1, 2, 3])
    data2 = tfield.preprocess([3, 2, 1])
    data3 = tfield.preprocess([3, 4, 5, 6])
    data4 = tfield.preprocess([2, 3, 6])

    tfield.finalize()

    assert np.all(tfield.numericalize(data1) == np.array([1, 2, 3]))
    assert np.all(tfield.numericalize(data2) == np.array([3, 2, 1]))
    assert np.all(tfield.numericalize(data3) == np.array([3, 4, 5, 6]))
    assert np.all(tfield.numericalize(data4) == np.array([2, 3, 6]))


def test_tokenized_field_custom_numericalization_2():
    label_indexer = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4
    }

    tfield = TokenizedField("bla",
                            custom_numericalize=label_indexer.get)

    data1 = tfield.preprocess(["one", "two", "three"])
    data2 = tfield.preprocess(["three", "two", "one"])
    data3 = tfield.preprocess(["three", "four", "four", "two"])
    data4 = tfield.preprocess(["two", "three", "one"])

    tfield.finalize()

    assert np.all(tfield.numericalize(data1) == np.array([1, 2, 3]))
    assert np.all(tfield.numericalize(data2) == np.array([3, 2, 1]))
    assert np.all(tfield.numericalize(data3) == np.array([3, 4, 4, 2]))
    assert np.all(tfield.numericalize(data4) == np.array([2, 3, 1]))


def test_tokenized_field_vocab_non_string():
    vocab = Vocab(specials=())
    tfield = TokenizedField("bla",
                            vocab=vocab)

    data1 = tfield.preprocess([1, 2, 3])
    data2 = tfield.preprocess([3, 2, 1])
    data3 = tfield.preprocess([3, 4, 5, 6])
    data4 = tfield.preprocess([2, 3, 6])

    tfield.finalize()

    assert np.all(tfield.numericalize(data1) == vocab.numericalize([1, 2, 3]))
    assert np.all(tfield.numericalize(data2) == vocab.numericalize([3, 2, 1]))
    assert np.all(tfield.numericalize(data3) == vocab.numericalize([3, 4, 5, 6]))
    assert np.all(tfield.numericalize(data4) == vocab.numericalize([2, 3, 6]))


def test_multilabel_field_specials_in_vocab_fail():
    with pytest.raises(ValueError):
        MultilabelField("bla",
                        vocab=Vocab(specials=(SpecialVocabSymbols.UNK,)),
                        num_of_classes=10)


@pytest.mark.parametrize("tokens",
                         [
                             ["class1", "class2", "class3", "class4"]
                         ])
def test_multilabel_field_vocab_numericalization(tokens):
    vocab = Vocab(specials=())
    vocab += tokens

    field = MultilabelField("test field", 5, vocab)
    preprocessed = field.preprocess(tokens)
    field.finalize()

    multilabel_from_vocab = np.zeros(5, dtype=np.bool)
    for token in tokens:
        multilabel_from_vocab[vocab.stoi[token]] = 1

    multilabel_from_field = field.numericalize(preprocessed)

    assert np.all(multilabel_from_field == multilabel_from_vocab)


@pytest.mark.parametrize("tokens, expected_numericalization",
                         [
                             (
                                     ["class1", "class2", "class3", "class4"],
                                     np.array([1, 1, 1, 1, 0, 0])
                             ),
                             (
                                     [],
                                     np.array([0, 0, 0, 0, 0, 0])
                             )
                         ])
def test_multilabel_field_custom_numericalization(tokens, expected_numericalization):
    index_dict = {
        "class1": 0,
        "class2": 1,
        "class3": 2,
        "class4": 3,
        "class5": 4,
        "class6": 5
    }

    field = MultilabelField("test field", 6, custom_numericalize=index_dict.get)
    preprocessed = field.preprocess(tokens)
    field.finalize()

    multilabel_from_field = field.numericalize(preprocessed)

    assert np.all(multilabel_from_field == expected_numericalization)


def test_multilabel_too_many_classes_in_data_exception():
    vocab = Vocab(specials=())
    field = MultilabelField("test_field", num_of_classes=3, vocab=vocab)

    for data in "cls1", "cls2", "cls3", "cls4":
        field.preprocess(data)

    with pytest.raises(ValueError):
        field.finalize()


@pytest.mark.parametrize("store_as_raw, store_as_tokenized, tokenize",
                         [[True, True, True],
                          [True, True, False],
                          [False, True, True],
                          [False, False, False]])
def test_field_fail_initialization(store_as_raw, store_as_tokenized, tokenize):
    with pytest.raises(ValueError):
        Field("bla",
              store_as_raw=store_as_raw,
              store_as_tokenized=store_as_tokenized,
              tokenize=tokenize)


def test_missing_values_default_sequential():
    fld = Field("bla",
                store_as_raw=False,
                tokenize=True,
                custom_numericalize=lambda x: hash(x),
                allow_missing_data=True)

    data_missing = fld.preprocess(None)
    data_exists = fld.preprocess("data_string")

    assert data_missing == (None, None)
    assert data_exists == (None, ["data_string"])
    fld.finalize()

    assert np.all(fld.numericalize(data_missing) == np.empty(0))
    assert np.all(fld.numericalize(data_exists) == np.array([hash("data_string")]))


def test_missing_values_default_not_sequential():
    fld = Field("bla",
                store_as_raw=True,
                tokenize=False,
                custom_numericalize=int,
                allow_missing_data=True)

    data_missing = fld.preprocess(None)
    data_exists = fld.preprocess("404")

    assert data_missing == (None, None)
    assert data_exists == ("404", None)

    fld.finalize()

    assert np.allclose(fld.numericalize(data_missing), np.array([np.nan]), equal_nan=True)
    assert np.all(fld.numericalize(data_exists) == np.array([404]))


def test_missing_values_fail():
    fld = Field("bla",
                store_as_raw=True,
                tokenize=False,
                custom_numericalize=lambda x: hash(x))

    with pytest.raises(ValueError):
        fld.preprocess(None)
