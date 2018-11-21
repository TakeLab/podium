import numpy as np
import pytest
from mock import patch

from takepod.storage.field import Field

ONE_TO_FIVE = [1, 2, 3, 4, 5]

CUSTOM_PAD = 43

PAD_NUM = 42


class MockVocab:
    def __init__(self):
        self.values = []
        self.finalized = False
        self.numericalized = False
        self.pad_symbol = PAD_NUM

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
        Field(name="F", store_raw=False, sequential=False)


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
    f = Field(name="F", store_raw=store_raw, sequential=sequential)

    received_raw_value, received_tokenized_value = f.preprocess(value)

    assert received_raw_value == expected_raw_value
    assert received_tokenized_value == expected_tokenized_value


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
              sequential=sequential)

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
    "use_vocab, expected_numericalized",
    [
        (False, False),
        (True, True),
    ]
)
def test_field_numericalize_vocab(use_vocab, expected_numericalized, vocab):
    f = Field(name="F", vocab=vocab if use_vocab else None, sequential=False)
    f.numericalize(("4.32", None))

    assert vocab.numericalized == expected_numericalized


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

    f = Field(name="F", vocab=vocab, tokenizer=my_tokenizer, sequential=True,
              store_raw=False)

    assert f.preprocess("asd dsa") == (None, ["a", "sd dsa"])


def test_field_get_tokenizer_spacy_exception(vocab):
    class MockSpacy:
        def load(self, x):
            raise OSError

    patch.dict("sys.modules", spacy=MockSpacy()).start()

    with pytest.raises(OSError):
        Field(name="F", vocab=vocab, tokenizer="spacy", sequential=True)


def test_field_get_tokenizer_default(vocab):
    f = Field(name="F", vocab=vocab, sequential=True, store_raw=False)

    assert f.preprocess("asd dsa") == (None, ["asd", "dsa"])


def test_field_get_tokenizer_exception(vocab):
    with pytest.raises(ValueError):
        Field(name="F", vocab=vocab, tokenizer="NOT_tokenizer",
              sequential=True, store_raw=False)


def test_field_get_tokenizer_spacy_ok(vocab):
    class MockToken:
        def __init__(self, txt):
            self.text = txt

    class MockSpacy:
        def load(self, x):
            class MockTokenizer:
                def tokenizer(self, string):
                    return [MockToken(tok) for tok in string.split()]

            return MockTokenizer()

    patch.dict("sys.modules", spacy=MockSpacy()).start()

    f = Field(name="F", vocab=vocab, tokenizer="spacy", sequential=True,
              store_raw=False)
    assert f.preprocess("bla blu") == (None, ["bla", "blu"])


def test_field_pretokenize_hooks():
    f = Field(name="F", sequential=True)

    f.add_pretokenize_hook(str.lower)
    f.add_pretokenize_hook(lambda x: x.replace("bla", "blu"))
    f.add_pretokenize_hook(lambda x: x.replace(";", " "))
    f.add_pretokenize_hook(lambda x: x.replace(",", " "))

    raw_str = "asd;123,BLA"

    received = f.preprocess(raw_str)
    expected = ("asd 123 blu", ["asd", "123", "blu"])

    assert received == expected


def test_field_pretokenize_hooks_detach():
    f = Field(name="F", sequential=True)

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
    f = Field(name="F", sequential=True)

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
    f = Field(name="F", sequential=True)

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

    f = Field(name="F", sequential=True)

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
