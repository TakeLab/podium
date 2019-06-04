import os
import tempfile
import shutil

from mock import patch

from takepod.dataload.eurovoc import EuroVocLoader, Label, LabelRank
from takepod.storage.large_resource import LargeResource
from takepod.storage.downloader import SCPDownloader

EUROVOC_LABELS = r"""
<DATABASE_THS>
    <RECORD>
        <Odrednica>thesaurus 1</Odrednica>
        <ID>000001</ID>
    </RECORD>
    <RECORD>
        <Odrednica>microthesaurus 1</Odrednica>
        <Podrucje>04;thesaurus 1</Podrucje>
        <ID>000002</ID>
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
        <Podrucje>04;thesaurus 1</Podrucje>
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


def test_creating_term_label():
    name = "label 1"
    label_id = 1
    direct_parents = [2]
    similar_terms = [3]
    rank = LabelRank.TERM
    thesaurus = 4
    micro_thesaurus = 5

    label = Label(name=name, id=label_id, direct_parents=direct_parents,
                  similar_terms=similar_terms, rank=rank, thesaurus=thesaurus,
                  micro_thesaurus=micro_thesaurus)

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

    label = Label(name=name, id=label_id, direct_parents=direct_parents,
                  similar_terms=similar_terms, rank=rank, thesaurus=thesaurus)

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

    label = Label(name=name, id=label_id, direct_parents=direct_parents,
                  similar_terms=similar_terms, rank=rank)

    assert label.name == name
    assert label.id == label_id
    assert label.direct_parents == direct_parents
    assert label.similar_terms == similar_terms
    assert label.rank == rank
    assert label.thesaurus is None
    assert label.micro_thesaurus is None


def create_mock_dataset(create_parent_dir=True):
    base_temp = tempfile.mkdtemp()
    assert os.path.exists(base_temp)

    if create_parent_dir:
        base_dataset_dir = os.path.join(base_temp, EuroVocLoader.NAME)
        os.makedirs(base_dataset_dir)
    else:
        base_dataset_dir = base_temp

    eurovoc_labels_path = os.path.join(base_dataset_dir,
                                       EuroVocLoader.EUROVOC_LABELS_FILENAME)
    create_file(eurovoc_labels_path, EUROVOC_LABELS)
    crovoc_labels_path = os.path.join(base_dataset_dir,
                                      EuroVocLoader.CROVOC_LABELS_FILENAME)
    create_file(crovoc_labels_path, CROVOC_LABELS)

    dataset_dir = os.path.join(base_dataset_dir,
                               EuroVocLoader.DATASET_DIR)
    os.makedirs(dataset_dir)
    document_1_path = os.path.join(dataset_dir,
                                   "NN00001.xml")
    create_file(document_1_path, DOCUMENT_1)

    document_2_path = os.path.join(dataset_dir,
                                   "NN00002.xml")
    create_file(document_2_path, DOCUMENT_2)

    document_3_path = os.path.join(dataset_dir,
                                   "NN00003.xml")
    create_file(document_3_path, INVALID_DOCUMENT)

    document_4_path = os.path.join(dataset_dir,
                                   "NN00004.xml")
    create_file(document_4_path, MISSING_DOCUMENT)

    mappings_path = os.path.join(base_dataset_dir,
                                 EuroVocLoader.MAPPING_FILENAME)

    mappings_content = ""
    with open("test/dataload/mock_mapping.xlsx", mode='rb') as input_file:
        mappings_content = input_file.read()
    with open(file=mappings_path, mode='wb') as fp:
        fp.write(mappings_content)

    return base_temp


def create_file(file_path, file_content):
    with open(file=file_path, mode='w', encoding="utf8") as fp:
        fp.writelines(file_content)


def test_loading_dataset():
    path = create_mock_dataset()
    LargeResource.BASE_RESOURCE_DIR = path
    loader = EuroVocLoader()
    eurovoc_labels, crovoc_labels, mappings, documents = loader.load_dataset()

    assert len(eurovoc_labels) == 4
    assert len(crovoc_labels) == 3
    assert len(mappings) == 3
    assert len(documents) == 2

    label_1 = eurovoc_labels[1]
    assert label_1.id == 1
    assert label_1.name == "thesaurus 1"
    assert label_1.rank == LabelRank.THESAURUS
    assert label_1.thesaurus == 1
    assert label_1.micro_thesaurus is None
    assert label_1.direct_parents == []
    assert label_1.similar_terms == []

    label_2 = eurovoc_labels[2]
    assert label_2.id == 2
    assert label_2.name == "microthesaurus 1"
    assert label_2.rank == LabelRank.MICRO_THESAURUS
    assert label_2.thesaurus == 1
    assert label_2.micro_thesaurus == 2
    assert label_2.direct_parents == [1]
    assert label_2.similar_terms == []

    label_3 = eurovoc_labels[3]
    assert label_3.id == 3
    assert label_3.name == "term 1"
    assert label_3.rank == LabelRank.TERM
    assert label_3.thesaurus == 1
    assert label_3.micro_thesaurus == 2
    assert label_3.direct_parents == [2]
    assert label_3.similar_terms == []

    label_4 = eurovoc_labels[4]
    assert label_4.micro_thesaurus is None
    assert label_4.direct_parents == [3]

    crovoc_label = crovoc_labels[13]
    assert crovoc_label.id == 13
    assert crovoc_label.name == "term cro 1"
    assert crovoc_label.rank == LabelRank.TERM
    assert crovoc_label.thesaurus == 11
    assert crovoc_label.micro_thesaurus == 12
    assert crovoc_label.direct_parents == [12]
    assert crovoc_label.similar_terms == []

    assert mappings[1] == [3, 13]
    assert mappings[2] == [4]

    assert documents[0].filename == "NN00001.xml"
    assert documents[0].title == "document title 1"
    assert documents[0].text == "document body 1"

    assert documents[1].filename == "NN00002.xml"
    assert documents[1].title == "document title 2"
    assert documents[1].text == "document body \n2"

    shutil.rmtree(path)
    assert not os.path.exists(path)


def mock_download(uri, path, overwrite=False, **kwargs):
    create_mock_zip_archive(path)


def create_mock_zip_archive(path):
    base = create_mock_dataset(create_parent_dir=False)
    shutil.make_archive(path, 'zip', root_dir=base, base_dir=None)
    shutil.move(path + ".zip", path)


@patch.object(SCPDownloader, 'download', mock_download)
def test_download_dataset_using_scp():
    LargeResource.BASE_RESOURCE_DIR = tempfile.mkdtemp()
    assert os.path.exists(LargeResource.BASE_RESOURCE_DIR)

    loader = EuroVocLoader(
        scp_user='username',
        scp_private_key='private_key',
        scp_pass_key='pass'
    )

    eurovoc_labels, crovoc_labels, mappings, documents = loader.load_dataset()
    assert len(eurovoc_labels) == 4
    assert len(crovoc_labels) == 3
    assert len(mappings) == 3
    assert len(documents) == 2
