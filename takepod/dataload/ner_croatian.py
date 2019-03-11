"""Simple NERCroatian dataset module."""
import glob
import os
import logging
import getpass
import xml.etree.ElementTree as ET

from takepod.preproc.tokenizers import get_tokenizer
from takepod.storage.large_resource import LargeResource, SCPLargeResource

_LOGGER = logging.getLogger(__name__)


class NERCroatianXMLLoader:
    """Simple croatian NER class"""

    URL = '/storage/takepod_data/datasets/CroatianNERDataset.zip'
    NAME = "CroatianNERDataset"
    SCP_HOST = "djurdja.takelab.fer.hr"
    ARCHIVE_TYPE = "zip"

    SENTENCE_DELIMITER_TOKEN = (None, None)

    def __init__(self,
                 path='downloaded_datasets/',
                 tokenizer='split',
                 tag_schema='IOB',
                 **kwargs):
        """
        Constructor for Croatian NER dataset.
        Downloads and extracts the dataset.

        Parameters
        ----------
        path: str
            Path to the folder where the dataset should be downloaded or loaded
            from if it is already downloaded
        tokenizer: str
            Word-level tokenizer used to tokenize the input text
        tag_schema: str
            Tag schema used for constructing the token labels
            - supported tag schemas:
                - 'IOB': the label of the beginning token of the entity is
                prefixed with 'B-', the remaining tokens that belong to the
                same entity are prefixed with 'I-'. The tokens that don't
                belong to any named entity are labeled 'O'
        **kwargs:
            scp_user:
                User on the host machine. Not required if the user on the
                local machine matches the user on the host machine.
            scp_private_key:
                Path to the ssh private key eligible to access the host
                machine. Not required on Unix if the private is in the default
                location.
            scp_pass_key:
                Password for the ssh private key (optional). Can be omitted
                if the private key is not encrypted.
        """
        self._data_dir = path
        self._tokenizer = get_tokenizer(tokenizer)
        self._label_resolver = self._get_label_resolver(tag_schema)

        if 'scp_user' not in kwargs:
            # if your username is same as one on djurdja
            scp_user = getpass.getuser()
        else:
            scp_user = kwargs['scp_user']

        scp_private_key = kwargs.get('scp_private_key', None)
        scp_pass_key = kwargs.get('scp_pass_key', None)

        config = {
            LargeResource.URI: NERCroatianXMLLoader.URL,
            LargeResource.RESOURCE_NAME: NERCroatianXMLLoader.NAME,
            LargeResource.ARCHIVE: NERCroatianXMLLoader.ARCHIVE_TYPE,
            SCPLargeResource.SCP_HOST_KEY: NERCroatianXMLLoader.SCP_HOST,
            SCPLargeResource.SCP_USER_KEY: scp_user,
            SCPLargeResource.SCP_PRIVATE_KEY: scp_private_key,
            SCPLargeResource.SCP_PASS_KEY: scp_pass_key
        }
        SCPLargeResource(**config)

    def load_dataset(self):
        """
        Method loads the dataset and returns tokenized NER documents.

        Returns
        -------
        tokenized_documents: list of lists of tuples
            List of tokenized documents. Each document is represented
            as a list of tuples (token, label). The sentences in document are
            delimited by tuple (None, None)
        """
        source_dir_location = os.path.join(self._data_dir,
                                           NERCroatianXMLLoader.NAME)

        tokenized_documents = []

        for xml_file_path in sorted(glob.glob(source_dir_location + '/*.xml')):
            word_label_pairs = self._xml_to_token_label_pairs(xml_file_path)
            tokenized_documents.append(word_label_pairs)

        return tokenized_documents

    def _xml_to_token_label_pairs(self, xml_file_path):
        """
        Converts the xml file located at the given path to the list of tuples
        (token, label)

        Parameters
        ----------
        xml_file_path: str
            Path to the XML file

        Returns
        -------
        token_label_pairs: list of tuples
            List of tuples (token, label)
        """
        root = ET.parse(xml_file_path).getroot()

        token_label_pairs = []

        for sentence in root.iter(tag='s'):
            for sub_element in sentence.iter():

                if sub_element.text is not None:
                    token_label_pairs_subelement = \
                        self._tokenize(sub_element.text.strip(), sub_element)
                    token_label_pairs.extend(token_label_pairs_subelement)

                if sub_element.tail is not None:
                    token_label_pairs_outside = \
                        self._tokenize(sub_element.tail.strip())
                    token_label_pairs.extend(token_label_pairs_outside)

            token_label_pairs.append(self.SENTENCE_DELIMITER_TOKEN)

        return token_label_pairs

    def _tokenize(self, text, element=None):
        """
        Method tokenizes the text and assigns the labels to the tokens
        according to the element's 'type' attribute.

        Parameters
        ----------
        text: str
            Input text
        element: ET.Element
            Element with which the text is associated.

        Returns
        -------
        token_label_pairs: list of tuples
            List of tuples (token, label)
        """
        if not text:
            return []

        tokenized_text = self._tokenizer(text)

        token_label_pairs = []
        for index, token in enumerate(tokenized_text):
            if element is not None:
                label_unprefixed = element.attrib.get('type', None)
                label = self._label_resolver(index, label_unprefixed)
            else:
                label = 'O'
            token_label_pairs.append((token, label))

        return token_label_pairs

    def _get_label_resolver(self, tag_schema):
        """
        Gets the label resolver associated with the given tag schema

        Parameters
        ----------
        tag_schema: str
            Tag schema for label prefixes

        Returns
        -------
        label_resolver: callable
            Label resolver associated with the given tag schema
        """
        if tag_schema == 'IOB':
            return self._iob_label_resolver

        error_msg = 'No label resolver for tag schema {} exists'\
                    .format(tag_schema)
        _LOGGER.error(error_msg)
        raise ValueError(error_msg)

    @staticmethod
    def _iob_label_resolver(index, label):
        """
        A resolver that prefixes the label according to the IOB tag schema.

        Parameters
        ----------
        index: int
            Index of the token/label in the named entity (starts from 0)
        label: str
            Label of the named entity

        Returns
        -------
            Label prefixed with the appropriate prefix according to the IOB
            tag schema
        """
        if label is None:
            return 'O'

        if index == 0:
            return 'B-' + label
        else:
            return 'I-' + label
