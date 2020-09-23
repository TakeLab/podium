import pytest
from podium.datasets import CornellMovieDialogsConversationalDataset
from podium.dataload.cornell_movie_dialogs import CornellMovieDialogsNamedTuple

EXPECTED_EXAMPLES = [
    {"statement": "They do not!".split(), "reply": "They do to!".split()},
    {"statement": "They do to!".split(), "reply": "I hope so.".split()},
    {"statement": "She okay?".split(), "reply": "Let's go.".split()},
    {"statement": "Wow".split(), "reply": "They do not!".split()},
    {"statement": "They do not!".split(), "reply": "They do to!".split()},
    {"statement": "They do to!".split(), "reply": "I hope so.".split()},
    {"statement": "She okay?".split(), "reply": "They do to!".split()}
]


def test_default_fields():
    fields = CornellMovieDialogsConversationalDataset.get_default_fields()
    assert len(fields) == 2
    field_names = ["statement", "reply"]
    assert all([name in fields for name in field_names])


def lines():
    return {
        "lineID": ["L194", "L195", "L196", "L198", "L199", "L200"],
        "characterID": ["u0", "u2", "u0", "u2", "u0", "u2"],
        "movieID": ["m0", "m0", "m0", "m0", "m0", "m0"],
        "character": ["BIANCA", "CAMERON", "BIANCA", "CAMERON", "BIANCA", "CAMERON"],
        "text": ["They do not!", "They do to!", "I hope so.", "She okay?", "Let's go.",
                 "Wow"]
    }


def conversations():
    return {"character1ID": ["u0", "u0", "u0", "u0", "u0"],
            "character2ID": ["u2", "u2", "u2", "u2", "u2"],
            "movieID": ["m0", "m0", "m0", "m0", "m0"],
            "utteranceIDs": [['L194', 'L195', 'L196', 'L197'], ['L198', 'L199'],
                             ['L200', 'L194', 'L195', 'L196'], ['L197', 'L198', 'L195'],
                             ['L199']]}


@pytest.fixture(scope="module")
def default_dataset():
    data = CornellMovieDialogsNamedTuple(titles=None, conversations=conversations(),
                                         lines=lines(), characters=None, url=None)
    return CornellMovieDialogsConversationalDataset(data=data)


def test_creating_dataset(default_dataset):
    dataset = default_dataset
    assert len(dataset) == 7

    for ex in dataset:
        ex_data = {"statement": ex.statement[1], "reply": ex.reply[1]}
        assert ex_data in EXPECTED_EXAMPLES


def test_data_none_error():
    with pytest.raises(ValueError):
        CornellMovieDialogsConversationalDataset(data=None)
