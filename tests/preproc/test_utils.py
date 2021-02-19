import pytest

from podium.preproc.utils import find_word_by_prefix, make_trie


def test_make_trie():
    words = ["hello", "hi"]
    trie = make_trie(words)
    expected_result = {"h": {"e": {"l": {"l": {"o": {"*": "*"}}}}, "i": {"*": "*"}}}
    assert trie == expected_result


@pytest.mark.parametrize(
    "input_word, expected",
    [
        ("hello", "hello"),
        ("hellosir", "hello"),
        ("h", None),
        ("bye", None),
        ("hella", None),
    ],
)
def test_find_word(input_word, expected):
    words = ["hello", "hi"]
    trie = make_trie(words)
    assert find_word_by_prefix(trie, input_word) == expected
