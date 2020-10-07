import pytest

import podium.preproc.stop_words as stop_words


@pytest.mark.parametrize(
    "example_raw, example_words, expected_result, stop_words_set",
    [
        (
            "Printer je bio prljav",
            ["Printer", "je", "bio", "prljav"],
            ["Printer", "prljav"],
            stop_words.CROATIAN,
        ),
        (
            "Barem je stol čist",
            ["Barem", "je", "stol", "Čist"],
            ["stol", "Čist"],
            stop_words.CROATIAN_EXTENDED,
        ),
    ],
)
def test_croatian_lemmatizer_hook(
    example_raw, example_words, expected_result, stop_words_set
):
    hook = stop_words.get_croatian_stop_words_removal_hook(stop_words_set=stop_words_set)
    result_raw, result_tokenized = hook(example_raw, example_words)
    assert result_tokenized == expected_result
    assert result_raw == example_raw


@pytest.mark.parametrize(
    "example_words, expected_result, stop_words_set",
    [
        (["Ako", "aj", "brtvA"], ["brtvA"], stop_words.CROATIAN_SPACY),
        (
            ["nepotpune", "BI", "vrijednosti"],
            ["nepotpune", "vrijednosti"],
            stop_words.CROATIAN,
        ),
    ],
)
def test_croatian_lemmatizer_hook_raw_none(
    example_words, expected_result, stop_words_set
):
    hook = stop_words.get_croatian_stop_words_removal_hook(stop_words_set=stop_words_set)
    result_raw, result_tokenized = hook(None, example_words)
    assert result_tokenized == expected_result
    assert result_raw is None
