import xml.etree.ElementTree as ET
import tempfile
import zipfile
import os
import shutil
import pytest

from mock import patch
from takepod.dataload.ner_croatian import (
    NERCroatianXMLLoader,
    convert_sequence_to_entities
)
from takepod.storage.resources.large_resource import LargeResource
from takepod.storage.resources.downloader import SCPDownloader


def create_ner_file(filepath, title_element, body_element):
    root = ET.Element("root")

    root.append(title_element)
    root.append(body_element)

    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8')


title_1 = ET.fromstring("""
<title>
    <s>Random <enamex type="Organization">Entitet</enamex></s>
</title>
""")

body_1 = ET.fromstring("<body><s>Ovdje nema entiteta!</s></body>")

expected_output_1 = [
    ('Random', 'O'),
    ('Entitet', 'B-Organization'),
    NERCroatianXMLLoader.SENTENCE_DELIMITER_TOKEN,
    ('Ovdje', 'O'),
    ('nema', 'O'),
    ('entiteta!', 'O'),
    NERCroatianXMLLoader.SENTENCE_DELIMITER_TOKEN
]

title_2 = ET.fromstring("""
<title>
    <s>
        <enamex type="LocationAsOrganization">Kina</enamex>
        je najveći svjetski izvoznik
    </s>
</title>
""")

body_2 = ET.fromstring("""
<body>
    <s>
        Ukupna vrijednost izvoza <timex type="Date">u prvoj polovini
        ove godine</timex> iznosila je <numex type="Money">521,7
        milijardi dolara</numex>.
    </s>
</body>
""")

expected_output_2 = [
    ('Kina', 'B-LocationAsOrganization'),
    ('je', 'O'),
    ('najveći', 'O'),
    ('svjetski', 'O'),
    ('izvoznik', 'O'),
    NERCroatianXMLLoader.SENTENCE_DELIMITER_TOKEN,
    ('Ukupna', 'O'),
    ('vrijednost', 'O'),
    ('izvoza', 'O'),
    ('u', 'B-Date'),
    ('prvoj', 'I-Date'),
    ('polovini', 'I-Date'),
    ('ove', 'I-Date'),
    ('godine', 'I-Date'),
    ('iznosila', 'O'),
    ('je', 'O'),
    ('521,7', 'B-Money'),
    ('milijardi', 'I-Money'),
    ('dolara', 'I-Money'),
    ('.', 'O'),
    NERCroatianXMLLoader.SENTENCE_DELIMITER_TOKEN
]


@pytest.mark.parametrize(
    "expected_data, expected_output",
    [
        ((title_1, body_1), expected_output_1),
        ((title_2, body_2), expected_output_2),
    ]
)
def test_load_dataset(tmpdir, expected_data, expected_output):
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    LargeResource.BASE_RESOURCE_DIR = base

    unzipped_xml_directory = os.path.join(
        base,
        NERCroatianXMLLoader.NAME
    )

    os.makedirs(unzipped_xml_directory)

    title, body = expected_data
    create_ner_file(
        os.path.join(unzipped_xml_directory, "file.xml"),
        title,
        body
    )

    ner_croatian_xml_loader = NERCroatianXMLLoader(
        base, tokenizer='split', tag_schema='IOB'
    )

    documents = ner_croatian_xml_loader.load_dataset()

    assert len(documents) == 1
    assert documents[0] == expected_output

    shutil.rmtree(base)
    assert not os.path.exists(base)


def test_load_dataset_with_multiple_documents():
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    LargeResource.BASE_RESOURCE_DIR = base

    unzipped_xml_directory = os.path.join(
        base,
        NERCroatianXMLLoader.NAME
    )

    os.makedirs(unzipped_xml_directory)

    create_ner_file(
        os.path.join(unzipped_xml_directory, "file_1.xml"),
        title_1,
        body_1
    )
    create_ner_file(
        os.path.join(unzipped_xml_directory, "file_2.xml"),
        title_2,
        body_2
    )

    ner_croatian_xml_loader = NERCroatianXMLLoader(
        base, tokenizer='split', tag_schema='IOB'
    )

    documents = ner_croatian_xml_loader.load_dataset()

    assert len(documents) == 2

    assert documents[0] == expected_output_1
    assert documents[1] == expected_output_2

    shutil.rmtree(base)
    assert not os.path.exists(base)


def test_load_dataset_with_unsupported_tokenizer():
    with pytest.raises(ValueError):
        NERCroatianXMLLoader(tokenizer='unsupported_tokenizer')


def test_load_dataset_with_unsupported_tag_schema():
    with pytest.raises(ValueError):
        NERCroatianXMLLoader(tag_schema='unsupported_tag_schema')


def mock_download(uri, path, overwrite=False, **kwargs):
    create_mock_zip_archive_with_xml_file(path)


@patch.object(SCPDownloader, 'download', mock_download)
def test_download_dataset_using_scp():
    base = tempfile.mkdtemp()
    assert os.path.exists(base)

    LargeResource.BASE_RESOURCE_DIR = base

    ner_croatian_xml_loader = NERCroatianXMLLoader(
        base,
        scp_user='username',
        scp_private_key='private_key',
        scp_pass_key='pass'
    )

    tokenized_documents = ner_croatian_xml_loader.load_dataset()

    assert len(tokenized_documents) == 1
    assert tokenized_documents[0] == expected_output_1

    shutil.rmtree(base)
    assert not os.path.exists(base)


@pytest.mark.parametrize(
    "sequence, text, expected_entities",
    [
        (
            [
                'B-Organization', 'I-Organization',
                'O', 'O',
                'B-Money', 'O', 'B-Organization'
            ],
            [
                'Kompanija', 'Microsoft',
                'je', 'kupila',
                '$8000', 'dionica', 'APIS-a'
            ],
            [
                {
                    'name': ['Kompanija', 'Microsoft'],
                    'type': 'Organization',
                    'start': 0,
                    'end': 2
                },
                {
                    'name': ['$8000'],
                    'type': 'Money',
                    'start': 4,
                    'end': 5
                },
                {
                    'name': ['APIS-a'],
                    'type': 'Organization',
                    'start': 6,
                    'end': 7
                }
            ]
        ),
        (
            [
                'O', 'B-Time', 'O', 'O',
                'B-Time', 'O', 'O', 'B-Test',
                'I-Test', 'I-Test', 'O'
            ],
            [
                'Jucer', 'popodne', 'je', 'bio',
                'dvotjedni', 'prosvjed', 'protiv',
                'testiranja', 'algoritamskih', 'zadataka',
                'natjecanja'
            ],
            [
                {
                    'name': ['popodne'],
                    'type': 'Time',
                    'start': 1,
                    'end': 2
                },
                {
                    'name': ['dvotjedni'],
                    'type': 'Time',
                    'start': 4,
                    'end': 5
                },
                {
                    'name': ['testiranja', 'algoritamskih', 'zadataka'],
                    'type': 'Test',
                    'start': 7,
                    'end': 10
                }
            ]
        ),
        (
            [
                'B-Organization', 'B-Organization',
                'O', 'O',
            ],
            [
                'Amazon', 'Microsoftu',
                'nije', 'suradnik'
            ],
            [
                {
                    'name': ['Amazon'],
                    'type': 'Organization',
                    'start': 0,
                    'end': 1
                },
                {
                    'name': ['Microsoftu'],
                    'type': 'Organization',
                    'start': 1,
                    'end': 2
                },
            ]
        ),
    ]
)
def test_convert_valid_sequence_to_entities(sequence, text, expected_entities):
    received_entities = convert_sequence_to_entities(sequence, text)
    assert received_entities == expected_entities


@pytest.mark.parametrize(
    "invalid_sequence, text, expected_entities",
    [
        # this example starts with "I" instead of "B"
        # resulting in ignoring this entity
        (
            [
                'I-Organization', 'O', 'O',
            ],
            [
                'Struji', 'struja', 'u'
            ],
            []
        ),
        # if tag description is different Organization!=Time
        # those tags are skipped
        (
            [
                'B-Organization', 'I-Time', 'O'
            ],
            [
                'FER', 'petak', 'je!'
            ],
            [
                {
                    'name': ['FER'],
                    'type': 'Organization',
                    'start': 0,
                    'end': 1
                }
            ]
        ),
    ]
)
def test_convert_invalid_sequence_to_entities(invalid_sequence, text, expected_entities):
    received_entities = convert_sequence_to_entities(invalid_sequence, text)
    assert received_entities == expected_entities


def test_invalid_delimiter():
    sequence = ['B', 'I', 'O', 'O', 'B']
    text = ['a', 'b', 'c', 'd', 'e']
    with pytest.raises(TypeError):
        convert_sequence_to_entities(sequence, text, delimiter=None)


def test_invalid_sequence_len():
    sequence = ['B', 'I']
    text = ['a']
    with pytest.raises(ValueError):
        convert_sequence_to_entities(sequence, text)


def create_mock_zip_archive_with_xml_file(dir_path):
    base = tempfile.mkdtemp()
    mock_xml_name = 'mock.xml'
    mock_xml_path = os.path.join(base, mock_xml_name)
    create_ner_file(mock_xml_path, title_1, body_1)

    with zipfile.ZipFile(file=dir_path, mode="w") as zipfp:
        zipfp.write(filename=mock_xml_path, arcname=mock_xml_name)

    shutil.rmtree(base)
    assert not os.path.exists(base)
