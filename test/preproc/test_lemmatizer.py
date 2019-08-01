import os
import pytest
import mock
from takepod.preproc.lemmatizer.croatian_lemmatizer import (
    CroatianLemmatizer, _lemmatizer_posttokenized_hook)


@pytest.mark.parametrize(
    "word, expected_lemma",
    [
        ("mami", "mama"),
        ("parkira", "parkirati"),
        ("mamama", "mama")
    ]
)
def test_word2lemma_base_case(word, expected_lemma, mock_lemmatizer):
    received_lemma = mock_lemmatizer.lemmatize_word(word)
    assert expected_lemma == received_lemma


@pytest.mark.parametrize(
    "word, expected_lemma",
    [
        ("MaMi", "MaMa"),  # base case, equal length
        ("parKira", "parKirati"),  # lemma longer than word
        ("Mamama", "Mama"),  # lemma shorter than word
        ("parkiranJe", "parkiranJe")  # uppercase on char after word
    ]
)
def test_word2lemma_casing(word, expected_lemma, mock_lemmatizer):
    received_lemma = mock_lemmatizer.lemmatize_word(word)
    assert expected_lemma == received_lemma


@pytest.mark.parametrize(
    "example_raw, example_words, expected_result",
    [
        ("MaMi parkira parkiranJe",
         ["MaMi", "parkira", "parkiranJe"],
         ["MaMa", "parkirati", "parkiranJe"]),
        ("tata Mamama",
         ["tata", "Mamama"],
         ["tata", "Mama"]),
    ]
)
def test_croatian_lemmatizer_hook(
        example_raw, example_words, expected_result, mock_lemmatizer):
    result_raw, result_tokenized = _lemmatizer_posttokenized_hook(
        raw=example_raw, tokenized=example_words, lemmatizer=mock_lemmatizer)
    assert result_tokenized == expected_result
    assert result_raw == example_raw


@pytest.mark.parametrize(
    "example_words, expected_result",
    [
        (["Mamama", "parkirati"], ["Mama", "parkirati"]),
        (["parkiranJe", "tatu"], ["parkiranJe", "tatu"]),
    ]
)
def test_croatian_lemmatizer_hook_raw_none(
        example_words, expected_result, mock_lemmatizer):
    result_raw, result_tokenized = _lemmatizer_posttokenized_hook(
        raw=None, tokenized=example_words, lemmatizer=mock_lemmatizer)
    assert result_tokenized == expected_result
    assert result_raw is None


@pytest.mark.parametrize(
    "lemma, expected_words",
    [
        ("mama", ["mama", "mame", "mami", "mamama",
                  "mamu", "mamo", "mamom", "mama"]),
        ("tata", ["tate", "tati", "tatata", "tatu"])
    ]
)
def test_lemma2word_base_case(lemma, expected_words, mock_lemmatizer):
    received_words = mock_lemmatizer.get_words_for_lemma(lemma)
    assert set(expected_words) == set(received_words)


@pytest.mark.parametrize(
    "lemma, expected_words",
    [
        ("maMa", ["maMa", "maMe", "maMi", "maMama",
                  "maMu", "maMo", "maMom", "maMa"]),

        # uppercase only if same letter at same position
        ("TatA", ["Tate", "Tati", "TatAta", "Tatu"])
    ]
)
def test_lemma2word_test_casing(lemma, expected_words, mock_lemmatizer):
    received_words = mock_lemmatizer.get_words_for_lemma(lemma)
    assert set(expected_words) == set(received_words)


def test_lemma2word_no_lemma_found(mock_lemmatizer):
    with pytest.raises(ValueError):
        mock_lemmatizer.get_words_for_lemma("not_in_molex")


def create_molex_file(filepath, content):
    with open(filepath, "w") as f:
        f.write(content)


@pytest.fixture()
def molex14_lemma2word(molexdir):
    content = ("mama#mame,mami,mamama,mamu,mamo,mamom,mama\n"
               "tata#tate,tati,tatata,tatu")
    path = os.path.join(molexdir, "molex14_lemma2word.txt")
    create_molex_file(path, content)
    return path


@pytest.fixture()
def molex14_word2lemma(molexdir):
    content = ("mamama mama\n"
               "mami mama\n"
               "parkira parkirati")
    path = os.path.join(molexdir, "molex14_word2lemma.txt")
    create_molex_file(path, content)
    return path


@pytest.fixture()
def molexdir(tmpdir):
    yield tmpdir


@pytest.fixture
def mock_lemmatizer(molex14_lemma2word, molex14_word2lemma):
    with mock.patch(
            'takepod.storage.resources.large_resource.SCPLargeResource.__init__'
    ) as mock_scp:
        mock_scp.return_value = None
        mock_scp.SCP_HOST_KEY = "scp_host"
        mock_scp.SCP_USER_KEY = "scp_user"
        mock_scp.SCP_PASS_KEY = "scp_pass"
        mock_scp.SCP_PRIVATE_KEY = "scp_priv"
        lemmatizer = CroatianLemmatizer()
        lemmatizer.MOLEX14_LEMMA2WORD = molex14_lemma2word
        lemmatizer.MOLEX14_WORD2LEMMA = molex14_word2lemma
        return lemmatizer


def test_lemmatize_oov(mock_lemmatizer):
    oov_word = 'outofvocabularyword'
    received_lemma = mock_lemmatizer.lemmatize_word(oov_word, none_if_oov=True)
    assert received_lemma is None
