"""Simple NERCroatian dataset module."""
import glob
import logging
import os
import xml.etree.ElementTree as ET

from podium.preproc.tokenizers import get_tokenizer
from podium.storage.resources.large_resource import init_scp_large_resource_from_kwargs
from podium.util import log_and_raise_error


_LOGGER = logging.getLogger(__name__)


class NERCroatianXMLLoader:
    """Simple croatian NER class"""

    URL = "/storage/takepod_data/datasets/CroatianNERDataset.zip"
    NAME = "CroatianNERDataset"
    SCP_HOST = "djurdja.takelab.fer.hr"
    ARCHIVE_TYPE = "zip"

    SENTENCE_DELIMITER_TOKEN = (None, None)

    def __init__(
        self, path="downloaded_datasets/", tokenizer="split", tag_schema="IOB", **kwargs
    ):
        """Constructor for Croatian NER dataset.
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
            SCPLargeResource.SCP_USER_KEY:
                User on the host machine. Not required if the user on the
                local machine matches the user on the host machine.
            SCPLargeResource.SCP_PRIVATE_KEY:
                Path to the ssh private key eligible to access the host
                machine. Not required on Unix if the private is in the default
                location.
            SCPLargeResource.SCP_PASS_KEY:
                Password for the ssh private key (optional). Can be omitted
                if the private key is not encrypted.
        """
        self._data_dir = path
        self._tokenizer = get_tokenizer(tokenizer)
        self._label_resolver = self._get_label_resolver(tag_schema)
        init_scp_large_resource_from_kwargs(
            resource=NERCroatianXMLLoader.NAME,
            uri=NERCroatianXMLLoader.URL,
            archive=NERCroatianXMLLoader.ARCHIVE_TYPE,
            user_dict=kwargs,
            scp_host=NERCroatianXMLLoader.SCP_HOST,
        )

    def load_dataset(self):
        """Method loads the dataset and returns tokenized NER documents.

        Returns
        -------
        tokenized_documents: list of lists of tuples
            List of tokenized documents. Each document is represented
            as a list of tuples (token, label). The sentences in document are
            delimited by tuple (None, None)
        """
        source_dir_location = os.path.join(self._data_dir, NERCroatianXMLLoader.NAME)

        tokenized_documents = []

        for xml_file_path in sorted(glob.glob(source_dir_location + "/*.xml")):
            word_label_pairs = self._xml_to_token_label_pairs(xml_file_path)
            tokenized_documents.append(word_label_pairs)

        return tokenized_documents

    def _xml_to_token_label_pairs(self, xml_file_path):
        """Converts the xml file located at the given path to the list of tuples
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

        for sentence in root.iter(tag="s"):
            for sub_element in sentence.iter():

                if sub_element.text is not None:
                    token_label_pairs_subelement = self._tokenize(
                        sub_element.text.strip(), sub_element
                    )
                    token_label_pairs.extend(token_label_pairs_subelement)

                if sub_element.tail is not None:
                    token_label_pairs_outside = self._tokenize(sub_element.tail.strip())
                    token_label_pairs.extend(token_label_pairs_outside)

            token_label_pairs.append(self.SENTENCE_DELIMITER_TOKEN)

        return token_label_pairs

    def _tokenize(self, text, element=None):
        """Method tokenizes the text and assigns the labels to the tokens
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
                label_unprefixed = element.attrib.get("type", None)
                label = self._label_resolver(index, label_unprefixed)
            else:
                label = "O"
            token_label_pairs.append((token, label))

        return token_label_pairs

    def _get_label_resolver(self, tag_schema):
        """Gets the label resolver associated with the given tag schema

        Parameters
        ----------
        tag_schema: str
            Tag schema for label prefixes

        Returns
        -------
        label_resolver: callable
            Label resolver associated with the given tag schema
        """
        if tag_schema == "IOB":
            return self._iob_label_resolver

        error_msg = f'No label resolver for tag schema {tag_schema} exists'
        log_and_raise_error(ValueError, _LOGGER, error_msg)

    @staticmethod
    def _iob_label_resolver(index, label):
        """A resolver that prefixes the label according to the IOB tag schema.

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
            return "O"
        elif index == 0:
            return "B-" + label
        return "I-" + label


def convert_sequence_to_entities(sequence, text, delimiter="-"):
    """Converts sequences of the BIO tagging schema to entities

    Parameters
    ----------
    sequence: list(string)
        Sequence of tags consisting that start with either B, I, or O.
    label: list(string)
        Tokenized text that correponds to the tag sequence

    Returns
    -------
    entities: list(dict)
        List of entities. Each entity is a dict that has four attributes:
        name, type, start, and end. Name is a list of tokens from text
        that belong to that entity, start denotes the index which starts
        the entity, and end is the end index of the entity.

        ```text[entity['start'] : entity['end']]``` retrieves the entity text

        Example
        {
            'name': list(str),
            'type': str,
            'start': int,
            'end': int
        }

    Raises
    ------
    ValueError
        If the given sequence and text are not of the same length.
    """
    entities = []
    state = "start"
    current_tag = "N/A"

    if len(text) != len(sequence):
        raise ValueError("Sequence and text must be of same length")

    for index, (tag, word) in enumerate(zip(sequence, text)):
        # must be either B, I, O
        if delimiter in tag:
            tag_type, tag_description = tag.split(delimiter)
        else:
            tag_type = tag[0]
            tag_description = ""

        if tag_type == "B" and state == "start":
            state = "named_entity"
            current_tag = tag_description
            # create new entity
            entity = {"name": [word], "type": tag_description, "start": index, "end": -1}
            entities.append(entity)

        elif tag_type == "B" and state == "named_entity":
            state = "named_entity"
            # save previous
            entities[-1]["end"] = index
            # create new one
            entity = {"name": [word], "type": tag_description, "start": index, "end": -1}
            entities.append(entity)

        elif tag_type == "I" and state == "named_entity":
            # I tag has to be after a B tag of the same type
            # B-Org I-Org is good, B-Org I-Time is not
            # I-Time part of the entity is skipped
            if tag_description == current_tag and entities:
                entities[-1]["name"].append(word)

            # if it does not match, just close the started entity
            elif tag_description != current_tag and entities:
                entities[-1]["end"] = index
                state = "start"

        elif tag_type == "O" and state == "named_entity":
            state = "start"
            if entities:
                entities[-1]["end"] = index

        elif tag_type == "O":
            state = "start"

    if entities and entities[-1]["end"] == -1:
        entities[-1]["end"] = len(sequence)

    return entities
