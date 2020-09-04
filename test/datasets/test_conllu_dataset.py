import os
import re
import tempfile

import pytest

from podium.datasets.impl.conllu_dataset import CoNLLUDataset


COMMENT = "# text = The quick brown fox jumps over the lazy dog."
DATA = """
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
"""  # noqa: E501


@pytest.fixture
def sample_dataset_raw():
    return '# text: ...\n' + re.sub(' +', '\t', DATA) + '\n'


def test_load_dataset(sample_dataset_raw):
    tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False)
    tmpfile.write(sample_dataset_raw)
    tmpfile.close()

    dataset = CoNLLUDataset(tmpfile.name)
    os.remove(tmpfile.name)

    assert len(dataset) == 10

    assert dataset[0].id[0] == '1'
    assert dataset[0].form[0] == 'The'
    assert dataset[0].lemma[0] == 'the'
    assert dataset[0].upos[0] == 'DET'
    assert dataset[0].xpos[0] == 'DT'
    assert dataset[0].feats[1] == [['Definite', 'Def'], ['PronType', 'Art']]
    assert dataset[0].head[0] == '4'
    assert dataset[0].deprel[0] == 'det'
    assert dataset[0].deps[1] is None
    assert dataset[0].misc[1] is None

    assert dataset[9].id[0] == '10'
    assert dataset[9].form[0] == '.'
    assert dataset[9].lemma[0] == '.'
    assert dataset[9].upos[0] == 'PUNCT'
    assert dataset[9].xpos[0] == '.'
    assert dataset[9].feats[1] is None
    assert dataset[9].head[0] == '5'
    assert dataset[9].deprel[0] == 'punct'
    assert dataset[9].deps[1] is None
    assert dataset[9].misc[1] is None


def test_default_fields():
    fields = CoNLLUDataset.get_default_fields()
    assert len(fields) == 10

    field_names = ['id', 'form', 'lemma', 'upos', 'xpos',
                   'feats', 'head', 'deprel', 'deps', 'misc']
    assert all([name in fields for name in field_names])

    assert not fields['id'].is_target
    assert not fields['form'].is_target
    assert not fields['lemma'].is_target
    assert not fields['feats'].is_target
    assert not fields['deps'].is_target
    assert not fields['misc'].is_target

    assert fields['upos'].is_target
    assert fields['xpos'].is_target
    assert fields['head'].is_target
    assert fields['deprel'].is_target
