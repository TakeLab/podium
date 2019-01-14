import xml.etree.ElementTree as ET
import tempfile
import zipfile
import pytest
import shutil
import os

from mock import patch
from takepod.dataload.ner_croatian import NERCroatianXMLLoader
from takepod.storage.large_resource import LargeResource
from takepod.storage.downloader import SCPDownloader


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

    unzipped_xml_directory = os.path.join(
        base,
        NERCroatianXMLLoader.NAME,
        NERCroatianXMLLoader.RESOURCE_NAME
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

    unzipped_xml_directory = os.path.join(
        base,
        NERCroatianXMLLoader.NAME,
        NERCroatianXMLLoader.RESOURCE_NAME
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


def create_mock_zip_archive_with_xml_file(dir_path):
    base = tempfile.mkdtemp()
    mock_xml_name = 'mock.xml'
    mock_xml_path = os.path.join(base, mock_xml_name)
    create_ner_file(mock_xml_path, title_1, body_1)

    with zipfile.ZipFile(file=dir_path, mode="w") as zipfp:
        zipfp.write(filename=mock_xml_path, arcname=mock_xml_name)

    shutil.rmtree(base)
    assert not os.path.exists(base)
