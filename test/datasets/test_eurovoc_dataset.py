import pytest
from mock import patch
from takepod.datasets.eurovoc_dataset import EuroVocDataset
from takepod.dataload.eurovoc import Label, LabelRank, Document
from takepod.preproc.lemmatizer.croatian_lemmatizer import CroatianLemmatizer


EXPECTED_EXAMPLES = [
    {"title": "title 1".split(),
     "text": "body text".split(),
     "eurovoc_labels": [3],
     "crovoc_labels": list()},
    {"title": "title 2".split(),
     "text": "body text".split(),
     "eurovoc_labels": [4],
     "crovoc_labels": [13]},
    {"title": "title 3".split(),
     "text": "body text".split(),
     "eurovoc_labels": [3, 4],
     "crovoc_labels": list()},
]

EXPECTED_EUROVOC_LABELS = [
    {"id": 1,
     "name": "thesaurus 1",
     "rank": LabelRank.THESAURUS,
     "micro_thesaurus": None,
     "thesaurus": 1,
     "direct_parents": [],
     "similar_terms": [],
     "all_ancestors": set()},
    {"id": 2,
     "name": "micro_thesaurus 2",
     "rank": LabelRank.MICRO_THESAURUS,
     "micro_thesaurus": 2,
     "thesaurus": 1,
     "direct_parents": [1],
     "similar_terms": [],
     "all_ancestors": {1}},
    {"id": 3,
     "name": "term 3",
     "rank": LabelRank.TERM,
     "micro_thesaurus": 2,
     "thesaurus": 1,
     "direct_parents": [2],
     "similar_terms": [],
     "all_ancestors": {1, 2}},
    {"id": 4,
     "name": "term 4",
     "rank": LabelRank.TERM,
     "micro_thesaurus": 2,
     "thesaurus": 1,
     "direct_parents": [2],
     "similar_terms": [],
     "all_ancestors": {1, 2}},
]

EXPECTED_CROVOC_LABELS = [
    {"id": 11,
     "name": "thesaurus 11",
     "rank": LabelRank.THESAURUS,
     "micro_thesaurus": None,
     "thesaurus": 11,
     "direct_parents": [],
     "similar_terms": [],
     "all_ancestors": set()},
    {"id": 12,
     "name": "micro_thesaurus 12",
     "rank": LabelRank.MICRO_THESAURUS,
     "micro_thesaurus": 12,
     "thesaurus": 11,
     "direct_parents": [11],
     "similar_terms": [],
     "all_ancestors": {11}},
    {"id": 13,
     "name": "term 13",
     "rank": LabelRank.TERM,
     "micro_thesaurus": 12,
     "thesaurus": 11,
     "direct_parents": [12],
     "similar_terms": [],
     "all_ancestors": {11, 12}},
]


def eurovoc_label_hierarchy():
    eurovoc_label_hierarchy = dict()
    label_1 = Label(name="thesaurus 1",
                    id=1,
                    rank=LabelRank.THESAURUS,
                    direct_parents=[],
                    similar_terms=[],
                    all_ancestors=set(),
                    thesaurus=1,
                    micro_thesaurus=None)
    eurovoc_label_hierarchy[1] = label_1
    label_2 = Label(name="micro_thesaurus 2",
                    id=2,
                    rank=LabelRank.MICRO_THESAURUS,
                    direct_parents=[1],
                    similar_terms=[],
                    all_ancestors={1},
                    thesaurus=1,
                    micro_thesaurus=2)
    eurovoc_label_hierarchy[2] = label_2
    label_3 = Label(name="term 3",
                    id=3,
                    rank=LabelRank.TERM,
                    direct_parents=[2],
                    similar_terms=[],
                    all_ancestors={1, 2},
                    thesaurus=1,
                    micro_thesaurus=2)
    eurovoc_label_hierarchy[3] = label_3
    label_4 = Label(name="term 4",
                    id=4,
                    rank=LabelRank.TERM,
                    direct_parents=[2],
                    similar_terms=[],
                    all_ancestors={1, 2},
                    thesaurus=1,
                    micro_thesaurus=2)
    eurovoc_label_hierarchy[4] = label_4
    return eurovoc_label_hierarchy


def crovoc_label_hierarchy():
    crovoc_label_hierarchy = dict()
    label_1 = Label(name="thesaurus 11",
                    id=11,
                    rank=LabelRank.THESAURUS,
                    direct_parents=[],
                    similar_terms=[],
                    all_ancestors=set(),
                    thesaurus=11,
                    micro_thesaurus=None)
    crovoc_label_hierarchy[11] = label_1
    label_2 = Label(name="micro_thesaurus 12",
                    id=12,
                    rank=LabelRank.MICRO_THESAURUS,
                    direct_parents=[11],
                    similar_terms=[],
                    all_ancestors={11},
                    thesaurus=11,
                    micro_thesaurus=12)
    crovoc_label_hierarchy[12] = label_2
    label_3 = Label(name="term 13",
                    id=13,
                    rank=LabelRank.TERM,
                    direct_parents=[12],
                    similar_terms=[],
                    all_ancestors={11, 12},
                    thesaurus=11,
                    micro_thesaurus=12)
    crovoc_label_hierarchy[13] = label_3
    return crovoc_label_hierarchy


def documents():
    documents = list()
    document_1 = Document("NN00100.xml",
                          "title 1",
                          "body text 1")
    documents.append(document_1)
    document_2 = Document("NN00200.xml",
                          "title 2",
                          "body text 2")
    documents.append(document_2)
    document_3 = Document("NN00300.xml",
                          "title 3",
                          "body text 3")
    documents.append(document_3)
    return documents


def missing_document():
    documents = list()
    document_1 = Document("NN00100.xml",
                          "title 1",
                          "body text 1")
    documents.append(document_1)
    document_2 = Document("NN00200.xml",
                          "title 2",
                          "body text 2")
    documents.append(document_2)
    return documents


def mappings():
    mappings = dict()
    mappings[100] = [3]
    mappings[200] = [4, 13]
    mappings[300] = [3, 4]
    return mappings


def mappings_missing_document():
    mappings = dict()
    mappings[100] = [3]
    mappings[200] = [4, 13]
    return mappings


def mappings_non_existing_label():
    mappings = dict()
    mappings[100] = [3, 5]
    mappings[200] = [4, 13]
    mappings[300] = [3, 4]
    return mappings


def mock_lemmatizer_posttokenized_hook(raw, tokenized, **kwargs):
    return (raw, tokenized)


def mock_init_lemmatizer(self, **kwargs):
    return


@patch('takepod.preproc.lemmatizer.croatian_lemmatizer._lemmatizer_posttokenized_hook',
       side_effect=mock_lemmatizer_posttokenized_hook)
@patch.object(CroatianLemmatizer, '__init__', mock_init_lemmatizer)
def test_default_fields(patched_hook):
    fields = EuroVocDataset.get_default_fields()
    assert len(fields) == 4
    field_names = ["text", "title", "eurovoc_labels", "crovoc_labels"]
    assert all([name in fields for name in field_names])


@pytest.fixture(scope="module")
@patch('takepod.preproc.lemmatizer.croatian_lemmatizer._lemmatizer_posttokenized_hook',
       side_effect=mock_lemmatizer_posttokenized_hook)
@patch.object(CroatianLemmatizer, '__init__', mock_init_lemmatizer)
def default_dataset(patched_hook):
    dataset = EuroVocDataset(eurovoc_labels=eurovoc_label_hierarchy(),
                             crovoc_labels=crovoc_label_hierarchy(),
                             mappings=mappings(),
                             documents=documents())
    return dataset


def test_creating_dataset(default_dataset):
    dataset = default_dataset
    assert len(dataset) == 3
    assert len(dataset.get_eurovoc_label_hierarchy()) == 4
    assert len(dataset.get_crovoc_label_hierarchy()) == 3


def test_created_examples(default_dataset):
    dataset = default_dataset
    for ex in dataset:
        ex_data = {"title": ex.title[1], "text": ex.text[1],
                   "eurovoc_labels": ex.eurovoc_labels[1],
                   "crovoc_labels": ex.crovoc_labels[1]}
        assert ex_data in EXPECTED_EXAMPLES


def test_loaded_eurovoc_labels(default_dataset):
    dataset = default_dataset
    eurovoc_labels = dataset.get_eurovoc_label_hierarchy()

    for label_id in eurovoc_labels:
        label = eurovoc_labels[label_id]
        label_data = {"id": label.id, "name": label.name, "rank": label.rank,
                      "micro_thesaurus": label.micro_thesaurus,
                      "thesaurus": label.thesaurus,
                      "direct_parents": label.direct_parents,
                      "similar_terms": label.similar_terms,
                      "all_ancestors": label.all_ancestors}
        assert label_data in EXPECTED_EUROVOC_LABELS


def test_loaded_crovoc_labels(default_dataset):
    dataset = default_dataset
    crovoc_labels = dataset.get_crovoc_label_hierarchy()

    for label_id in crovoc_labels:
        label = crovoc_labels[label_id]
        label_data = {"id": label.id, "name": label.name, "rank": label.rank,
                      "micro_thesaurus": label.micro_thesaurus,
                      "thesaurus": label.thesaurus,
                      "direct_parents": label.direct_parents,
                      "similar_terms": label.similar_terms,
                      "all_ancestors": label.all_ancestors}
        assert label_data in EXPECTED_CROVOC_LABELS


@patch('takepod.preproc.lemmatizer.croatian_lemmatizer._lemmatizer_posttokenized_hook',
       side_effect=mock_lemmatizer_posttokenized_hook)
@patch.object(CroatianLemmatizer, '__init__', mock_init_lemmatizer)
def test_missing_document(patched_hook):
    dataset = EuroVocDataset(eurovoc_labels=eurovoc_label_hierarchy(),
                             crovoc_labels=crovoc_label_hierarchy(),
                             mappings=mappings(),
                             documents=missing_document())
    assert len(dataset) == 2
    assert len(dataset.get_eurovoc_label_hierarchy()) == 4
    assert len(dataset.get_crovoc_label_hierarchy()) == 3


@patch('takepod.preproc.lemmatizer.croatian_lemmatizer._lemmatizer_posttokenized_hook',
       side_effect=mock_lemmatizer_posttokenized_hook)
@patch.object(CroatianLemmatizer, '__init__', mock_init_lemmatizer)
def test_missing_document_mapping(patched_hook):
    dataset = EuroVocDataset(eurovoc_labels=eurovoc_label_hierarchy(),
                             crovoc_labels=crovoc_label_hierarchy(),
                             mappings=mappings_missing_document(),
                             documents=documents())
    assert len(dataset) == 2
    assert len(dataset.get_eurovoc_label_hierarchy()) == 4
    assert len(dataset.get_crovoc_label_hierarchy()) == 3


@patch('takepod.preproc.lemmatizer.croatian_lemmatizer._lemmatizer_posttokenized_hook',
       side_effect=mock_lemmatizer_posttokenized_hook)
@patch.object(CroatianLemmatizer, '__init__', mock_init_lemmatizer)
def test_non_existing_label(patched_hook):
    dataset = EuroVocDataset(eurovoc_labels=eurovoc_label_hierarchy(),
                             crovoc_labels=crovoc_label_hierarchy(),
                             mappings=mappings_non_existing_label(),
                             documents=documents())

    assert len(dataset) == 3
    assert len(dataset.get_eurovoc_label_hierarchy()) == 4
    assert len(dataset.get_crovoc_label_hierarchy()) == 3

    for ex in dataset:
        ex_data = {"title": ex.title[1], "text": ex.text[1],
                   "eurovoc_labels": ex.eurovoc_labels[1],
                   "crovoc_labels": ex.crovoc_labels[1]}
        assert ex_data in EXPECTED_EXAMPLES


def test_is_ancestor(default_dataset):
    dataset = default_dataset

    for example in dataset:
        assert dataset.is_ancestor(1, example) is True
        assert dataset.is_ancestor(2, example) is True
        assert dataset.is_ancestor(13, example) is False


def test_get_parents(default_dataset):
    dataset = default_dataset

    assert dataset.get_direct_parents(1) == []
    assert dataset.get_direct_parents(2) == [1]
    assert dataset.get_direct_parents(3) == [2]

    assert dataset.get_direct_parents(11) == []
    assert dataset.get_direct_parents(12) == [11]
    assert dataset.get_direct_parents(13) == [12]


def test_get_parents_of_non_existing_label(default_dataset):
    dataset = default_dataset

    assert dataset.get_direct_parents(15) is None


def test_get_all_ancestors(default_dataset):
    dataset = default_dataset

    assert dataset.get_all_ancestors(1) == set()
    assert dataset.get_all_ancestors(2) == {1}
    assert dataset.get_all_ancestors(3) == {1, 2}

    assert dataset.get_all_ancestors(11) == set()
    assert dataset.get_all_ancestors(12) == {11}
    assert dataset.get_all_ancestors(13) == {11, 12}


def test_get_ancestors_of_non_existing_label(default_dataset):
    dataset = default_dataset

    assert dataset.get_all_ancestors(15) is None
