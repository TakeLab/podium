import pytest

from podium.preproc.sentencizers import SpacySentencizer

from ..util import has_spacy_model, is_admin


RUN_SPACY = is_admin or has_spacy_model("en")


@pytest.mark.skipif(
    not RUN_SPACY,
    reason="requires already downloaded model or "
    "admin privileges to download it "
    "while executing",
)
@pytest.mark.parametrize(
    "test_sentences,split_on",
    [
        ("Sentence 1. Sentence 2.", "."),
        ("Sentence 1! Sentence 2.", "!"),
        ("Sentence 1? Sentence 2.", "?"),
    ],
)
def test_spacy_sentencizer(test_sentences, split_on):
    sentences = test_sentences.split(split_on, maxsplit=1)
    sentences[0] += split_on
    sentences[1] = sentences[1].lstrip()
    assert SpacySentencizer("en")(test_sentences) == sentences
