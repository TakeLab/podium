import os
import re
import tempfile

import pytest

from podium.datasets.impl.conllu_dataset import CoNLLUDataset


DATA = """
# text = The quick brown fox jumps over the lazy dog.
1   The     the    DET    DT   Definite=Def|PronType=Art   4   det     _   _
2   quick   quick  ADJ    JJ   Degree=Pos                  4   amod    _   _
3   brown   brown  ADJ    JJ   Degree=Pos                  4   amod    _   _
4   fox     fox    NOUN   NN   Number=Sing                 5   nsubj   _   _
5   jumps   jump   VERB   VBZ  Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin   0   root    _   _
6   over    over   ADP    IN   _                           9   case    _   _
7   the     the    DET    DT   Definite=Def|PronType=Art   9   det     _   _
8   lazy    lazy   ADJ    JJ   Degree=Pos                  9   amod    _   _
9   dog     dog    NOUN   NN   Number=Sing                 5   nmod    _   SpaceAfter=No
10  .       .      PUNCT  .    _                           5   punct   _   _

# text = The quick
1   The     the    DET    DT   Definite=Def|PronType=Art   4   det     _   _
2   quick   quick  ADJ    JJ   Degree=Pos                  4   amod    _   _
"""  # noqa: E501
FAULTY_DATA = """
wrong_id_format   The     the    DET    DT   Definite=Def|PronType=Art   4   det     _   _
"""


@pytest.fixture
def sample_dataset_raw():
    return re.sub(" +", "\t", DATA) + "\n"


@pytest.fixture
def sample_dataset_raw_faulty():
    return re.sub(" +", "\t", FAULTY_DATA) + "\n"


def test_load_dataset(sample_dataset_raw):
    pytest.importorskip("conllu")

    tmpfile = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpfile.write(sample_dataset_raw)
    tmpfile.close()

    dataset = CoNLLUDataset(tmpfile.name)
    os.remove(tmpfile.name)

    assert len(dataset) == 2

    example_0 = dataset[0]
    example_1 = dataset[1]

    assert example_0["id"][1] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert example_0["form"][1] == [
        "The",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
        ".",
    ]
    assert example_0["lemma"][1] == [
        "the",
        "quick",
        "brown",
        "fox",
        "jump",
        "over",
        "the",
        "lazy",
        "dog",
        ".",
    ]
    assert example_0["upos"][1] == [
        "DET",
        "ADJ",
        "ADJ",
        "NOUN",
        "VERB",
        "ADP",
        "DET",
        "ADJ",
        "NOUN",
        "PUNCT",
    ]
    assert example_0["xpos"][1] == [
        "DT",
        "JJ",
        "JJ",
        "NN",
        "VBZ",
        "IN",
        "DT",
        "JJ",
        "NN",
        ".",
    ]
    assert example_0["feats"][1] == [
        {"Definite": "Def", "PronType": "Art"},
        {"Degree": "Pos"},
        {"Degree": "Pos"},
        {"Number": "Sing"},
        {
            "Mood": "Ind",
            "Number": "Sing",
            "Person": "3",
            "Tense": "Pres",
            "VerbForm": "Fin",
        },
        None,
        {"Definite": "Def", "PronType": "Art"},
        {"Degree": "Pos"},
        {"Number": "Sing"},
        None,
    ]
    assert example_0["head"][1] == [4, 4, 4, 5, 0, 9, 9, 9, 5, 5]
    assert example_0["deprel"][1] == [
        "det",
        "amod",
        "amod",
        "nsubj",
        "root",
        "case",
        "det",
        "amod",
        "nmod",
        "punct",
    ]
    assert example_0["deps"][1] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert example_0["misc"][1] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {"SpaceAfter": "No"},
        None,
    ]

    assert example_1["id"][1] == [1, 2]
    assert example_1["form"][1] == ["The", "quick"]
    assert example_1["lemma"][1] == ["the", "quick"]
    assert example_1["upos"][1] == ["DET", "ADJ"]
    assert example_1["xpos"][1] == ["DT", "JJ"]
    assert example_1["feats"][1] == [
        {"Definite": "Def", "PronType": "Art"},
        {"Degree": "Pos"},
    ]
    assert example_1["head"][1] == [4, 4]
    assert example_1["deprel"][1] == ["det", "amod"]
    assert example_1["deps"][1] == [None, None]
    assert example_1["misc"][1] == [None, None]


def test_load_faulty_dataset(sample_dataset_raw_faulty):
    pytest.importorskip("conllu")

    tmpfile = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpfile.write(sample_dataset_raw_faulty)
    tmpfile.close()

    with pytest.raises(ValueError):
        CoNLLUDataset(tmpfile.name)

    os.remove(tmpfile.name)
