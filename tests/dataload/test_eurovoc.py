import os
import shutil
import tempfile
from unittest.mock import patch

import dill
import pytest

from podium.dataload.eurovoc import EuroVocLoader, Label, LabelRank, dill_dataset
from podium.datasets.impl.eurovoc_dataset import EuroVocDataset
from podium.storage import Field, MultilabelField, Vocab
from podium.storage.resources.large_resource import LargeResource, SCPLargeResource

from ..datasets.test_eurovoc_dataset import (
    crovoc_label_hierarchy,
    documents,
    eurovoc_label_hierarchy,
    mappings,
)


EUROVOC_LABELS = r"""
<DATABASE_THS>
    <RECORD>
        <Odrednica>thesaurus 1</Odrednica>
        <ID>000001</ID>
    </RECORD>
    <RECORD>
        <Odrednica>term 1</Odrednica>
        <Podrucje>04;thesaurus 1</Podrucje>
        <Potpojmovnik>microthesaurus 1</Potpojmovnik>
        <ID>000003</ID>
    </RECORD>
    <RECORD>
        <Odrednica>term 2</Odrednica>
        <Podrucje>04;thesaurus 1</Podrucje>
        <Potpojmovnik>microthesaurus 2</Potpojmovnik>
        <SiriPojam>term 1</SiriPojam>
        <SrodniPojam>term 1</SrodniPojam>
        <ID>000004</ID>
    </RECORD>
    <RECORD>
        <Odrednica>microthesaurus 1</Odrednica>
        <Podrucje>04;thesaurus 1</Podrucje>
        <ID>000002</ID>
    </RECORD>
</DATABASE_THS>
"""

CROVOC_LABELS = r"""
<DATABASE_THS>
    <RECORD>
        <Odrednica>thesaurus cro 1</Odrednica>
        <ID>000011</ID>
    </RECORD>
    <RECORD>
        <Odrednica>microthesaurus cro 1</Odrednica>
        <Podrucje>04;thesaurus cro 1</Podrucje>
        <ID>000012</ID>
    </RECORD>
    <RECORD>
        <Odrednica>term cro 1</Odrednica>
        <Podrucje>04;thesaurus cro 1</Podrucje>
        <Potpojmovnik>microthesaurus cro 1</Potpojmovnik>
        <ID>000013</ID>
    </RECORD>
</DATABASE_THS>
"""

INVALID_LABELS = r"""
<DATABASE_THS>
    <RECORD>
        <Odrednica>thesaurus 1</Odrednica>
        <Potpojmovnik>microthesaurus 2</Potpojmovnik>
        <ID>000001</ID>
    </RECORD>
</DATABASE_THS>
"""

NON_EXISTING_THESAURUS = r"""
<DATABASE_THS>
    <RECORD>
        <Odrednica>term 1</Odrednica>
        <Podrucje>04;thesaurus 1</Podrucje>
        <Potpojmovnik>microthesaurus 1</Potpojmovnik>
        <ID>000003</ID>
    </RECORD>
</DATABASE_THS>
"""

DOCUMENT_1 = r"""
<doc>
    <HEAD>
        <TITLE>1 25.01.2010 Document title 1</TITLE>
    </HEAD>
    <BODY>Document body 1</BODY>
</doc>
"""

DOCUMENT_2 = r"""
<doc>
    <HEAD>
        <TITLE>2 25.01.2010 Document title 2</TITLE>
    </HEAD>
    <BODY>Document body 2</BODY>
</doc>
"""

DOCUMENT_2_BR = r"""
<doc>
    <HEAD>
        <TITLE>2 25.01.2010 Document title 2</TITLE>
    </HEAD>
    <BODY>Document body <br></br>2</BODY>
</doc>
"""

INVALID_DOCUMENT = r"""
<doc>
    <HEAD>
        <TITLE>3 25.01.2010 Document title 3</TITLE>
    </HEAD>
    <BODY>
        Document body 3
        postupak ekstrakcije teksta
    </BODY>
</doc>
"""

MISSING_DOCUMENT = r"""
<doc>
    <HEAD>
        <TITLE>4 25.01.2010 Document title 4</TITLE>
    </HEAD>
    <BODY>Document body 4</BODY>
</doc>
"""

INVALID_DOCUMENT_TITLE = r"""
<doc>
    <HEAD>
        <TITLE> HIDRA </TITLE>
    </HEAD>
    <BODY>Document body 5</BODY>
</doc>
"""


def test_creating_term_label():
    name = "label 1"
    label_id = 1
    direct_parents = [2]
    similar_terms = [3]
    rank = LabelRank.TERM
    thesaurus = 4
    micro_thesaurus = 5

    label = Label(
        name=name,
        id=label_id,
        direct_parents=direct_parents,
        similar_terms=similar_terms,
        rank=rank,
        thesaurus=thesaurus,
        micro_thesaurus=micro_thesaurus,
    )

    assert label.name == name
    assert label.id == label_id
    assert label.direct_parents == direct_parents
    assert label.similar_terms == similar_terms
    assert label.rank == rank
    assert label.thesaurus == thesaurus
    assert label.micro_thesaurus == micro_thesaurus


def test_creating_microthesaurus_label():
    name = "label 1"
    label_id = 1
    direct_parents = [2]
    similar_terms = [3]
    rank = LabelRank.MICRO_THESAURUS
    thesaurus = 4

    label = Label(
        name=name,
        id=label_id,
        direct_parents=direct_parents,
        similar_terms=similar_terms,
        rank=rank,
        thesaurus=thesaurus,
    )

    assert label.name == name
    assert label.id == label_id
    assert label.direct_parents == direct_parents
    assert label.similar_terms == similar_terms
    assert label.rank == rank
    assert label.thesaurus == thesaurus
    assert label.micro_thesaurus is None


def test_creating_thesaurus_label():
    name = "label 1"
    label_id = 1
    direct_parents = [2]
    similar_terms = [3]
    rank = LabelRank.THESAURUS

    label = Label(
        name=name,
        id=label_id,
        direct_parents=direct_parents,
        similar_terms=similar_terms,
        rank=rank,
    )

    assert label.name == name
    assert label.id == label_id
    assert label.direct_parents == direct_parents
    assert label.similar_terms == similar_terms
    assert label.rank == rank
    assert label.thesaurus is None
    assert label.micro_thesaurus is None


def create_mock_dataset(
    load_missing_doc=False,
    load_invalid_doc=False,
    load_doc_with_br_tag=False,
    load_doc_with_invalid_title=False,
    invalid_labels=False,
    non_existing_thesaurus=False,
    create_parent_dir=True,
):
    base_temp = tempfile.mkdtemp()
    assert os.path.exists(base_temp)

    if create_parent_dir:
        base_dataset_dir = os.path.join(base_temp, EuroVocLoader.NAME)
        os.makedirs(base_dataset_dir)
    else:
        base_dataset_dir = base_temp

    eurovoc_labels_path = os.path.join(
        base_dataset_dir, EuroVocLoader.EUROVOC_LABELS_FILENAME
    )
    if invalid_labels:
        create_file(eurovoc_labels_path, INVALID_LABELS)
    elif non_existing_thesaurus:
        create_file(eurovoc_labels_path, NON_EXISTING_THESAURUS)
    else:
        create_file(eurovoc_labels_path, EUROVOC_LABELS)

    crovoc_labels_path = os.path.join(
        base_dataset_dir, EuroVocLoader.CROVOC_LABELS_FILENAME
    )
    create_file(crovoc_labels_path, CROVOC_LABELS)

    dataset_dir = os.path.join(base_dataset_dir, EuroVocLoader.DATASET_DIR)
    os.makedirs(dataset_dir)
    document_1_path = os.path.join(dataset_dir, "NN00001.xml")
    create_file(document_1_path, DOCUMENT_1)

    document_2_path = os.path.join(dataset_dir, "NN00002.xml")
    if load_doc_with_br_tag:
        create_file(document_2_path, DOCUMENT_2_BR)
    else:
        document_2_path = os.path.join(dataset_dir, "NN00002.xml")
        create_file(document_2_path, DOCUMENT_2)

    if load_invalid_doc:
        document_3_path = os.path.join(dataset_dir, "NN00003.xml")
        create_file(document_3_path, INVALID_DOCUMENT)

    if load_missing_doc:
        document_4_path = os.path.join(dataset_dir, "NN00004.xml")
        create_file(document_4_path, MISSING_DOCUMENT)

    if load_doc_with_invalid_title:
        document_5_path = os.path.join(dataset_dir, "NN00005.xml")
        create_file(document_5_path, INVALID_DOCUMENT_TITLE)

    mappings_path = os.path.join(base_dataset_dir, EuroVocLoader.MAPPING_FILENAME)

    mappings_content = ""
    with open("tests/dataload/mock_mapping.xlsx", mode="rb") as input_file:
        mappings_content = input_file.read()
    with open(file=mappings_path, mode="wb") as fp:
        fp.write(mappings_content)

    return base_temp


def create_file(file_path, file_content):
    with open(file=file_path, mode="w", encoding="utf8") as fp:
        fp.writelines(file_content)


def test_loading_dataset():
    path = create_mock_dataset()
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", path):
        loader = EuroVocLoader()
        eurovoc_labels, crovoc_labels, mappings, documents = loader.load_dataset()

        assert len(eurovoc_labels) == 4
        assert len(crovoc_labels) == 3
        assert len(mappings) == 4
        assert len(documents) == 2
        documents.sort(key=lambda doc: doc.filename)

        label_1 = eurovoc_labels[1]
        assert label_1.id == 1
        assert label_1.name == "thesaurus 1"
        assert label_1.rank == LabelRank.THESAURUS
        assert label_1.thesaurus == 1
        assert label_1.micro_thesaurus is None
        assert label_1.direct_parents == []
        assert label_1.all_ancestors == set()
        assert label_1.similar_terms == []

        label_2 = eurovoc_labels[2]
        assert label_2.id == 2
        assert label_2.name == "microthesaurus 1"
        assert label_2.rank == LabelRank.MICRO_THESAURUS
        assert label_2.thesaurus == 1
        assert label_2.micro_thesaurus == 2
        assert label_2.direct_parents == [1]
        assert label_2.all_ancestors == {1}
        assert label_2.similar_terms == []

        label_3 = eurovoc_labels[3]
        assert label_3.id == 3
        assert label_3.name == "term 1"
        assert label_3.rank == LabelRank.TERM
        assert label_3.thesaurus == 1
        assert label_3.micro_thesaurus == 2
        assert label_3.direct_parents == [2]
        assert label_3.all_ancestors == {1, 2}
        assert label_3.similar_terms == []

        label_4 = eurovoc_labels[4]
        assert label_4.micro_thesaurus is None
        assert label_4.direct_parents == [3]
        assert label_4.all_ancestors == {1, 2, 3}
        assert label_4.similar_terms == [3]

        crovoc_label = crovoc_labels[13]
        assert crovoc_label.id == 13
        assert crovoc_label.name == "term cro 1"
        assert crovoc_label.rank == LabelRank.TERM
        assert crovoc_label.thesaurus == 11
        assert crovoc_label.micro_thesaurus == 12
        assert crovoc_label.direct_parents == [12]
        assert crovoc_label.similar_terms == []

        crovoc_label_1 = crovoc_labels[11]
        assert crovoc_label_1.id == 11
        assert crovoc_label_1.name == "thesaurus cro 1"
        assert crovoc_label_1.rank == LabelRank.THESAURUS
        assert crovoc_label_1.thesaurus == 11
        assert crovoc_label_1.micro_thesaurus is None
        assert crovoc_label_1.direct_parents == []
        assert crovoc_label_1.similar_terms == []

        crovoc_label_2 = crovoc_labels[12]
        assert crovoc_label_2.id == 12
        assert crovoc_label_2.name == "microthesaurus cro 1"
        assert crovoc_label_2.rank == LabelRank.MICRO_THESAURUS
        assert crovoc_label_2.thesaurus == 11
        assert crovoc_label_2.micro_thesaurus == 12
        assert crovoc_label_2.direct_parents == [11]
        assert crovoc_label_2.similar_terms == []

        assert mappings[1] == [3, 13]
        assert mappings[2] == [4]
        assert mappings[3] == [3, 13]
        assert mappings[5] == [3]

        assert documents[0].filename == "NN00001.xml"
        assert documents[0].title == "document title 1"
        assert documents[0].text == "document body 1"

        assert documents[1].filename == "NN00002.xml"
        assert documents[1].title == "document title 2"
        assert documents[1].text == "document body 2"

        shutil.rmtree(path)
        assert not os.path.exists(path)


def test_loading_dataset_with_missing_document():
    path = create_mock_dataset(load_missing_doc=True)
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", path):
        loader = EuroVocLoader()
        eurovoc_labels, crovoc_labels, mappings, documents = loader.load_dataset()

        assert len(eurovoc_labels) == 4
        assert len(crovoc_labels) == 3
        assert len(mappings) == 4
        assert len(documents) == 2
        documents.sort(key=lambda doc: doc.filename)

        assert documents[0].filename == "NN00001.xml"
        assert documents[1].filename == "NN00002.xml"

        shutil.rmtree(path)
        assert not os.path.exists(path)


def test_loading_dataset_with_invalid_document():
    path = create_mock_dataset(load_invalid_doc=True)
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", path):
        loader = EuroVocLoader()
        eurovoc_labels, crovoc_labels, mappings, documents = loader.load_dataset()

        assert len(eurovoc_labels) == 4
        assert len(crovoc_labels) == 3
        assert len(mappings) == 4
        assert len(documents) == 2
        documents.sort(key=lambda doc: doc.filename)

        assert documents[0].filename == "NN00001.xml"
        assert documents[1].filename == "NN00002.xml"

        shutil.rmtree(path)
        assert not os.path.exists(path)


def test_loading_dataset_with_document_containing_br():
    path = create_mock_dataset(load_doc_with_br_tag=True)
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", path):
        loader = EuroVocLoader()
        eurovoc_labels, crovoc_labels, mappings, documents = loader.load_dataset()

        assert len(eurovoc_labels) == 4
        assert len(crovoc_labels) == 3
        assert len(mappings) == 4
        assert len(documents) == 2
        documents.sort(key=lambda doc: doc.filename)

        assert documents[0].filename == "NN00001.xml"
        assert documents[1].filename == "NN00002.xml"
        assert documents[1].title == "document title 2"
        assert documents[1].text == "document body \n2"

        shutil.rmtree(path)
        assert not os.path.exists(path)


def test_loading_dataset_with_invalid_title():
    path = create_mock_dataset(load_doc_with_invalid_title=True)
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", path):
        loader = EuroVocLoader()
        eurovoc_labels, crovoc_labels, mappings, documents = loader.load_dataset()

        assert len(eurovoc_labels) == 4
        assert len(crovoc_labels) == 3
        assert len(mappings) == 4
        assert len(documents) == 3
        documents.sort(key=lambda doc: doc.filename)

        assert documents[0].filename == "NN00001.xml"
        assert documents[1].filename == "NN00002.xml"
        assert documents[2].filename == "NN00005.xml"

        assert documents[2].title == "hidra"
        assert documents[2].text == "document body 5"

        shutil.rmtree(path)
        assert not os.path.exists(path)


def test_loading_dataset_with_invalid_labels():
    with pytest.raises(ValueError):
        path = create_mock_dataset(invalid_labels=True)
        with patch.object(LargeResource, "BASE_RESOURCE_DIR", path):
            loader = EuroVocLoader()
            loader.load_dataset()


def test_loading_dataset_with_non_existing_thesaurus():
    path = create_mock_dataset(non_existing_thesaurus=True)
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", path):
        loader = EuroVocLoader()
        eurovoc_labels, _, _, _ = loader.load_dataset()

        label = eurovoc_labels[3]
        assert label.thesaurus is None


def mock_download(self):
    assert self.config[SCPLargeResource.SCP_USER_KEY] == "username"
    assert self.config[SCPLargeResource.SCP_PRIVATE_KEY] == "private_key"
    assert self.config[SCPLargeResource.SCP_PASS_KEY] == "pass"


@patch.object(LargeResource, "_download_unarchive", mock_download)
def test_download_dataset_using_scp():
    base = tempfile.mkdtemp()
    with patch.object(LargeResource, "BASE_RESOURCE_DIR", base):
        assert os.path.exists(LargeResource.BASE_RESOURCE_DIR)

        EuroVocLoader(
            **{
                SCPLargeResource.SCP_USER_KEY: "username",
                SCPLargeResource.SCP_PRIVATE_KEY: "private_key",
                SCPLargeResource.SCP_PASS_KEY: "pass",
            }
        )


def mock_init(*args):
    return None


def mock_load_dataset(*args):
    return eurovoc_label_hierarchy(), crovoc_label_hierarchy(), mappings(), documents()


def mock_get_default_fields():
    title = Field(
        name="title", vocab=Vocab(), tokenizer="split", language="hr", tokenize=True
    )
    text = Field(
        name="text", vocab=Vocab(keep_freqs=True), tokenizer="split", tokenize=True
    )
    labels = MultilabelField(name="eurovoc_labels", vocab=Vocab(specials=()))
    crovoc_labels = MultilabelField(name="crovoc_labels", vocab=Vocab(specials=()))
    fields = {
        "title": title,
        "text": text,
        "eurovoc_labels": labels,
        "crovoc_labels": crovoc_labels,
    }
    return fields


@patch.object(EuroVocLoader, "__init__", mock_init)
@patch.object(EuroVocLoader, "load_dataset", mock_load_dataset)
@patch.object(EuroVocDataset, "get_default_fields", mock_get_default_fields)
def test_dill_dataset(tmpdir):
    path = os.path.join(tmpdir, "dataset.dill")
    dill_dataset(path)
    with open(path, "rb") as input_file:
        dataset = dill.load(input_file)

    assert len(dataset) == 3
    assert len(dataset.get_eurovoc_label_hierarchy()) == 4
    assert len(dataset.get_crovoc_label_hierarchy()) == 3
    assert len(dataset.fields) == 4