import os
from unittest.mock import Mock, patch

import dill
import numpy as np
import pytest

from podium.storage import (
    Field,
    LabelField,
    MultilabelField,
    MultioutputField,
    SpecialVocabSymbols,
    Vocab,
)


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


class MockVocab(Mock):
    def __init__(self, eager=True):
        super(MockVocab, self).__init__(spec=Vocab)
        self.values = []
        self.finalized = False
        self.numericalized = False
        self.eager = eager

    def padding_index(self):
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

    def __getitem__(self, item):
        raise NotImplementedError()


@pytest.mark.skip(reason="obsolete as the combination no longer raises an error")
def test_field_store_raw_sequential_exception():
    with pytest.raises(ValueError):
        Field(name="F", store_as_raw=False, tokenize=False)


def test_field_preprocess_eager():
    vocab = MockVocab(eager=True)
    f = Field(name="F", numericalizer=vocab)
    f.preprocess("some text")

    # vocab was updated
    assert len(vocab.values) > 0


@pytest.mark.parametrize(
    "value, store_raw, tokenize, expected_raw_value, " "expected_tokenized_value",
    [
        ("some text", True, True, "some text", ["some", "text"]),
        ("some text", True, False, "some text", "some text"),
        ("some text", False, True, None, ["some", "text"]),
        ("some text", False, False, None, "some text"),
    ],
)
def test_field_preprocess_raw_sequential(
    value, store_raw, tokenize, expected_raw_value, expected_tokenized_value
):
    tokenizer = "split" if tokenize else None
    f = Field(name="F", keep_raw=store_raw, tokenizer=tokenizer)

    ((_, (received_raw_value, received_tokenized_value)),) = f.preprocess(value)

    assert received_raw_value == expected_raw_value
    assert received_tokenized_value == expected_tokenized_value


@pytest.mark.parametrize(
    "value, store_raw, tokenize, expected_raw_value, " "expected_tokenized_value",
    [
        ("some text", True, True, "some text", ["some", "text"]),
        ("some text", True, False, "some text", "some text"),
        ("some text", False, True, None, ["some", "text"]),
    ],
)
def test_field_pickle_tokenized(
    value, store_raw, tokenize, expected_raw_value, expected_tokenized_value, tmpdir
):
    tokenizer = "split" if tokenize else None
    fld = Field(name="F", keep_raw=store_raw, tokenizer=tokenizer)

    ((_, (received_raw_value, received_tokenized_value)),) = fld.preprocess(value)

    assert received_raw_value == expected_raw_value
    assert received_tokenized_value == expected_tokenized_value

    field_file = os.path.join(tmpdir, "field.pkl")

    with open(field_file, "wb") as fdata:
        dill.dump(fld, fdata)

    with open(field_file, "rb") as fdata:
        loaded_fld = dill.load(fdata)
        ((_, (raw_value, tokenized_value)),) = loaded_fld.preprocess(value)

        assert raw_value == expected_raw_value
        assert tokenized_value == expected_tokenized_value
        assert loaded_fld.name == "F"
        assert loaded_fld.keep_raw == store_raw


@pytest.mark.parametrize("vocab, expected_value", [(MockVocab(), True), (None, False)])
def test_field_use_vocab(vocab, expected_value):
    f = Field(name="F", numericalizer=vocab)

    assert f.use_vocab == expected_value


def test_field_finalize():
    vocab = MockVocab()
    f = Field(name="F", numericalizer=vocab)

    assert not vocab.finalized
    f.finalize()
    assert vocab.finalized
    with pytest.raises(Exception):
        f.finalize()


def test_field_custom_numericalize():
    f = Field(name="F", numericalizer=float)
    numericalized = f.numericalize((None, "4.32"))

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
        (ONE_TO_FIVE, 0, [], False, True),
    ],
)
def test_field_pad_to_length(row, length, expected_row, pad_left, truncate_left):
    vocab = MockVocab()
    f = Field(name="F", numericalizer=vocab)

    received_row = f.pad_to_length(
        np.array(row), length, pad_left=pad_left, truncate_left=truncate_left
    )

    assert received_row.tolist() == expected_row


def test_field_pad_custom_numericalize():
    custom_padding_token = -999
    f = Field(
        "test_field",
        numericalizer=int,
        padding_token=custom_padding_token,
        tokenizer="split",
    )
    mock_numericalization = np.array([1, 2, 3, 4])
    expected_numericalization = np.array([1, 2, 3, 4] + [custom_padding_token] * 6)

    padded = f.pad_to_length(mock_numericalization, 10, pad_left=False)
    assert np.all(padded == expected_numericalization)


@pytest.mark.parametrize(
    "row, length, expected_row",
    [
        (ONE_TO_FIVE, 3, [1, 2, 3]),
        (ONE_TO_FIVE, 7, [1, 2, 3, 4, 5, CUSTOM_PAD, CUSTOM_PAD]),
        (ONE_TO_FIVE, 5, ONE_TO_FIVE),
        (ONE_TO_FIVE, 0, []),
    ],
)
def test_field_pad_to_length_custom_pad(row, length, expected_row):
    f = Field(name="F", numericalizer=None)

    row_arr = np.array(row)
    received_row = f.pad_to_length(row_arr, length, custom_pad_symbol=CUSTOM_PAD)

    assert received_row.tolist() == expected_row


def test_field_pad_to_length_exception():
    # set vocab to be None
    f = Field(name="F", numericalizer=None, padding_token=None)

    row_arr = np.array(ONE_TO_FIVE)
    length = 7

    custom_pad_symbol = None
    with pytest.raises(ValueError):
        f.pad_to_length(row_arr, length, custom_pad_symbol=custom_pad_symbol)


def test_field_get_tokenizer_callable():
    vocab = MockVocab()

    def my_tokenizer(string):
        return [string[0], string[1:]]

    f = Field(name="F", numericalizer=vocab, tokenizer=my_tokenizer)

    _, data = f.preprocess("asd dsa")[0]
    assert data == (None, ["a", "sd dsa"])


@pytest.mark.skip(
    "obsolete as the combination no longer raises an error, "
    "see https://github.com/mttk/podium/pull/180 for more information"
)
def test_field_get_tokenizer_spacy_exception():
    # vocab = MockVocab()

    class MockSpacy:
        def load(self, x, **kwargs):
            raise OSError

    patch.dict("sys.modules", spacy=MockSpacy()).start()


#     with pytest.raises(OSError):
#         Field(name="F", numericalizer=vocab, tokenizer="spacy", tokenize=True)


def test_field_get_tokenizer_default():
    f = Field(name="F", numericalizer=MockVocab())

    _, data = f.preprocess("asd dsa")[0]
    assert data == (None, ["asd", "dsa"])


def test_field_get_tokenizer_exception():
    with pytest.raises(ValueError):
        Field(name="F", numericalizer=MockVocab(), tokenizer="NOT_tokenizer")


def test_field_get_tokenizer_spacy_ok():
    patch.dict("sys.modules", spacy=MockSpacy()).start()
    f = Field(name="F", numericalizer=MockVocab(), tokenizer="spacy")
    _, data = f.preprocess("bla blu")[0]
    assert data == (None, ["bla", "blu"])


def test_field_pickle_spacy_tokenizer(tmpdir):
    patch.dict("sys.modules", spacy=MockSpacy()).start()
    fld = Field(name="F", numericalizer=None, tokenizer="spacy")
    _, data = fld.preprocess("bla blu")[0]
    assert data == (None, ["bla", "blu"])

    field_file = os.path.join(tmpdir, "field.pkl")

    with open(field_file, "wb") as fdata:
        dill.dump(fld, fdata)

    with open(field_file, "rb") as fdata:
        loaded_fld = dill.load(fdata)

        assert loaded_fld._tokenizer_arg_string == "spacy"

        _, data = loaded_fld.preprocess("bla blu")[0]
        assert data == (None, ["bla", "blu"])


def test_field_pretokenize_hooks():
    f = Field(name="F", tokenizer="split", keep_raw=True)

    f.add_pretokenize_hook(str.lower)
    f.add_pretokenize_hook(lambda x: x.replace("bla", "blu"))
    f.add_pretokenize_hook(lambda x: x.replace(";", " "))
    f.add_pretokenize_hook(lambda x: x.replace(",", " "))

    raw_str = "asd;123,BLA"

    _, received = f.preprocess(raw_str)[0]
    expected = ("asd 123 blu", ["asd", "123", "blu"])

    assert received == expected


def test_field_pretokenize_hooks_detach():
    f = Field(name="F", tokenizer="split", keep_raw=True)

    f.add_pretokenize_hook(str.lower)
    f.add_pretokenize_hook(lambda x: x.replace(";", " "))
    f.add_pretokenize_hook(lambda x: x.replace(",", " "))

    # detaching
    f.remove_pretokenize_hooks()

    raw_str = "asd;123,BLA"

    _, received = f.preprocess(raw_str)[0]

    expected = (raw_str, [raw_str])

    assert received == expected


def test_field_posttokenize_hooks():
    f = Field(name="F", tokenizer="split", keep_raw=True)

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

    _, received = f.preprocess("asd 123<tag> B<tag>LA")[0]
    expected = ("ASD 123 BLA", ["ASD", "123", "BLA"])

    assert received == expected


def test_field_posttokenize_hooks_detach():
    f = Field(name="F", tokenizer="split", numericalizer=float, keep_raw=True)

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

    _, received = f.preprocess("asd 123<tag> B<tag>LA")[0]
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

    f = Field(name="F", tokenizer="split", numericalizer=float, keep_raw=True)

    # TAG -> tag
    f.add_posttokenize_hook(to_lower_hook)

    # <tag> -> ABC
    f.add_posttokenize_hook(replace_tag_hook)

    # ABC -> abc
    f.add_posttokenize_hook(to_lower_hook)

    _, received = f.preprocess("BLA <TAG> bla")[0]

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


def test_field_vocab_no_tokenization():
    vocab = Vocab()
    pretokenized_input1 = ["word", "words", "uttering"]
    pretokenized_input2 = ["word", "words"]
    pretokenized_input3 = ["word"]

    pretokenized_input4 = ["word", "uttering"]

    tokenized_field = Field("test_field", tokenizer=None, numericalizer=vocab)

    _, data1 = tokenized_field.preprocess(pretokenized_input1)[0]
    _, data2 = tokenized_field.preprocess(pretokenized_input2)[0]
    _, data3 = tokenized_field.preprocess(pretokenized_input3)[0]
    _, data4 = tokenized_field.preprocess(pretokenized_input4)[0]

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


def test_field_custom_numericalization_no_tokenization():
    tfield = Field("bla", numericalizer=lambda x: x, tokenizer=None)

    _, data1 = tfield.preprocess([1, 2, 3])[0]
    _, data2 = tfield.preprocess([3, 2, 1])[0]
    _, data3 = tfield.preprocess([3, 4, 5, 6])[0]
    _, data4 = tfield.preprocess([2, 3, 6])[0]

    tfield.finalize()

    assert np.all(tfield.numericalize(data1) == np.array([1, 2, 3]))
    assert np.all(tfield.numericalize(data2) == np.array([3, 2, 1]))
    assert np.all(tfield.numericalize(data3) == np.array([3, 4, 5, 6]))
    assert np.all(tfield.numericalize(data4) == np.array([2, 3, 6]))


def test_field_custom_numericalization_no_tokenization_2():
    label_indexer = {"one": 1, "two": 2, "three": 3, "four": 4}

    tfield = Field("bla", numericalizer=label_indexer.get, tokenizer=None)

    _, data1 = tfield.preprocess(["one", "two", "three"])[0]
    _, data2 = tfield.preprocess(["three", "two", "one"])[0]
    _, data3 = tfield.preprocess(["three", "four", "four", "two"])[0]
    _, data4 = tfield.preprocess(["two", "three", "one"])[0]

    tfield.finalize()

    assert np.all(tfield.numericalize(data1) == np.array([1, 2, 3]))
    assert np.all(tfield.numericalize(data2) == np.array([3, 2, 1]))
    assert np.all(tfield.numericalize(data3) == np.array([3, 4, 4, 2]))
    assert np.all(tfield.numericalize(data4) == np.array([2, 3, 1]))


def test_field_custom_numericalization_vocab_non_string():
    vocab = Vocab(specials=())
    tfield = Field("bla", numericalizer=vocab, tokenizer=None)

    _, data1 = tfield.preprocess([1, 2, 3])[0]
    _, data2 = tfield.preprocess([3, 2, 1])[0]
    _, data3 = tfield.preprocess([3, 4, 5, 6])[0]
    _, data4 = tfield.preprocess([2, 3, 6])[0]

    tfield.finalize()

    assert np.all(tfield.numericalize(data1) == vocab.numericalize([1, 2, 3]))
    assert np.all(tfield.numericalize(data2) == vocab.numericalize([3, 2, 1]))
    assert np.all(tfield.numericalize(data3) == vocab.numericalize([3, 4, 5, 6]))
    assert np.all(tfield.numericalize(data4) == vocab.numericalize([2, 3, 6]))


def test_multilabel_field_specials_in_vocab_fail():
    with pytest.raises(ValueError):
        MultilabelField(
            name="bla",
            numericalizer=Vocab(specials=(SpecialVocabSymbols.UNK,)),
            num_of_classes=10,
        )


@pytest.mark.parametrize("tokens", [["class1", "class2", "class3", "class4"]])
def test_multilabel_field_vocab_numericalization(tokens):
    vocab = Vocab(specials=())
    vocab += tokens

    field = MultilabelField("test field", num_of_classes=5, numericalizer=vocab)
    ((_, preprocessed),) = field.preprocess(tokens)
    field.finalize()

    multilabel_from_vocab = np.zeros(5, dtype=np.bool)
    for token in tokens:
        multilabel_from_vocab[vocab.stoi[token]] = 1

    multilabel_from_field = field.numericalize(preprocessed)

    assert np.all(multilabel_from_field == multilabel_from_vocab)


def test_multilabel_field_class_count():
    vocab = Vocab(specials=())
    field = MultilabelField(name="test field", num_of_classes=None, numericalizer=vocab)

    example_1 = ["class1", "class2", "class3", "class4"]
    example_2 = ["class1", "class2", "class3"]

    ((_, data_1),) = field.preprocess(example_1)
    ((_, data_2),) = field.preprocess(example_2)
    field.finalize()

    assert field.num_of_classes == 4

    numericalized = field.numericalize(data_1)
    assert len(numericalized) == 4

    numericalized = field.numericalize(data_2)
    assert len(numericalized) == 4


@pytest.mark.parametrize(
    "tokens, expected_numericalization",
    [
        (["class1", "class2", "class3", "class4"], np.array([1, 1, 1, 1, 0, 0])),
        ([], np.array([0, 0, 0, 0, 0, 0])),
    ],
)
def test_multilabel_field_custom_numericalization(tokens, expected_numericalization):
    index_dict = {
        "class1": 0,
        "class2": 1,
        "class3": 2,
        "class4": 3,
        "class5": 4,
        "class6": 5,
    }

    field = MultilabelField(
        name="test field", num_of_classes=6, numericalizer=index_dict.get, tokenizer=None
    )
    ((_, preprocessed),) = field.preprocess(tokens)
    field.finalize()

    multilabel_from_field = field.numericalize(preprocessed)

    assert np.all(multilabel_from_field == expected_numericalization)


def test_multilabel_too_many_classes_in_data_exception():
    vocab = Vocab(specials=())
    field = MultilabelField(name="test_field", num_of_classes=3, numericalizer=vocab)

    for data in "cls1", "cls2", "cls3", "cls4":
        field.preprocess(data)

    with pytest.raises(ValueError):
        field.finalize()


def test_missing_values_default_sequential():
    fld = Field(
        name="bla",
        keep_raw=False,
        tokenizer="split",
        numericalizer=hash,
        allow_missing_data=True,
    )

    _, data_missing = fld.preprocess(None)[0]
    _, data_exists = fld.preprocess("data_string")[0]

    assert data_missing == (None, None)
    assert data_exists == (None, ["data_string"])
    fld.finalize()

    assert fld.numericalize(data_missing) is None
    assert np.all(fld.numericalize(data_exists) == np.array([hash("data_string")]))


def test_missing_values_custom_numericalize():
    fld = Field(
        name="test_field",
        keep_raw=True,
        tokenizer=None,
        numericalizer=int,
        allow_missing_data=True,
    )

    _, data_missing = fld.preprocess(None)[0]
    _, data_exists = fld.preprocess("404")[0]

    assert data_missing == (None, None)
    assert data_exists == ("404", "404")

    fld.finalize()

    assert fld.numericalize(data_missing) is None
    assert np.all(fld.numericalize(data_exists) == np.array([404]))


def test_missing_symbol_index_vocab():
    vocab = Vocab()
    fld = Field(
        name="test_field",
        tokenizer="split",
        keep_raw=False,
        numericalizer=vocab,
        allow_missing_data=True,
    )

    fld.preprocess("a b c d")
    ((_, data),) = fld.preprocess(None)
    assert data == (None, None)

    fld.finalize()
    assert fld.numericalize((None, None)) is None
    assert fld.get_default_value() == -1


def test_missing_symbol_index_custom_numericalize():
    fld = Field(
        name="test_field",
        keep_raw=True,
        tokenizer=None,
        numericalizer=int,
        allow_missing_data=True,
    )

    fld.finalize()
    assert fld.get_default_value() == -1


def test_missing_values_fail():
    fld = Field(name="bla", keep_raw=True, tokenizer=None, numericalizer=hash)

    with pytest.raises(ValueError):
        fld.preprocess(None)


def test_multioutput_field_posttokenization():
    uppercase_field = Field("uppercase_field", keep_raw=True)
    lowercase_field = Field("lowercase_field", keep_raw=True)

    def post_tokenization_all_upper(raw, tokenized):
        return raw, list(map(str.upper, tokenized))

    def post_tokenization_all_lower(raw, tokenized):
        return raw, list(map(str.lower, tokenized))

    uppercase_field.add_posttokenize_hook(post_tokenization_all_upper)
    lowercase_field.add_posttokenize_hook(post_tokenization_all_lower)

    output_fields = uppercase_field, lowercase_field
    mo_field = MultioutputField(output_fields, tokenizer="split")

    result1, result2 = mo_field.preprocess("mOcK TeXt")

    assert result1 == (uppercase_field.name, ("mOcK TeXt", ["MOCK", "TEXT"]))
    assert result2 == (lowercase_field.name, ("mOcK TeXt", ["mock", "text"]))


def test_multioutput_field_remove_pretokenization():
    output_field_1 = Field("test_field_1")
    output_field_2 = Field("test_field_2")

    def first_lower(raw, tokenized):
        def f(token):
            if len(token) == 0:
                return ""
            else:
                return token[0].lower() + token[1:]

        return raw, list(map(f, tokenized))

    output_field_2.add_posttokenize_hook(first_lower)

    mo_field = MultioutputField((output_field_1, output_field_2))
    mo_field.add_pretokenize_hook(str.upper)

    (_, (raw_1, tokenized_1)), (_, (raw_2, tokenized_2)) = mo_field.preprocess(
        "this is a test sentence"
    )

    assert tokenized_1 == ["THIS", "IS", "A", "TEST", "SENTENCE"]
    assert tokenized_2 == ["tHIS", "iS", "a", "tEST", "sENTENCE"]


def test_posttokenize_hooks_in_field_no_tokenization_single_execution(mocker):
    f = Field(name="F", tokenizer=None)

    def hk(data, tokenized):
        def caseness(token):
            if token.islower():
                return "lowercase"
            else:
                return "uppercase"

        return data, list(map(caseness, tokenized))

    patched_hook = mocker.spy(hk, "__call__")

    f.add_posttokenize_hook(patched_hook)

    raw_str = ["Upper", "lower"]

    _, received = f.preprocess(raw_str)[0]
    expected = (None, ["uppercase", "lowercase"])

    assert received == expected
    patched_hook.assert_called_once()


def test_hook_returning_iterable():
    data = "1,2,3,4"
    expected_tokens = [3, 5, 7, 9]

    field = Field(
        "Iterator_hook_test_field",
        tokenizer=lambda raw: [int(x) for x in raw.split(",")],
        numericalizer=id,
        keep_raw=True,
    )

    def multiply_by_two_hook(raw, tokens):
        return raw, (i * 2 for i in tokens)

    def add_one_hook(raw, tokens):
        assert not isinstance(tokens, (list, tuple))
        return raw, (i + 1 for i in tokens)

    field.add_posttokenize_hook(multiply_by_two_hook)
    field.add_posttokenize_hook(add_one_hook)

    _, (raw, tokens) = field.preprocess(data)[0]

    assert raw == data
    assert isinstance(tokens, (list, tuple))
    assert tokens == expected_tokens


def test_label_field():
    vocab = Vocab(specials=())
    data = ["label_1", "label_2", "label_3"]

    vocab += data
    vocab.finalize()

    label_field = LabelField("test_label_field", numericalizer=vocab)

    preprocessed_data = [label_field.preprocess(label) for label in data]

    label_field.finalize()

    for x in preprocessed_data:
        _, data = x[0]
        _, tokenized = data
        assert label_field.numericalize(data) == vocab.stoi[tokenized]
