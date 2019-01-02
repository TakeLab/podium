"""Simple NERCroatian dataset module."""
import glob
import os
import xml.etree.ElementTree as ET


class NERCroatianXMLLoader:
    """Simple croatian NER class"""

    SENTENCE_DELIMITER_TOKEN = (None, None)

    def __init__(self,
                 path='downloaded_datasets/',
                 tokenizer='split',
                 tag_schema='IOB'):
        """Constructor for Croatian NER dataset

        Parameters
        ----------
        path: str
            Path to the folder where the dataset should be downloaded or loaded
            from if it is already downloaded
        tokenizer: str
            Word-level tokenizer used to tokenize the input text
            - supported tokenizers:
                - 'split': simple tokenizer that splits the sentence on
                    whitespaces (using str.split)
        tag_schema: str
            Tag schema used for constructing the token labels
            - supported tag schemas:
                - 'IOB': the label of the beginning token of the entity is
                prefixed with 'B-', the remaining tokens that belong to the
                same entity are prefixed with 'I-'. The tokens that don't
                belong to any named entity are labeled 'O'
        """
        self._data_dir = path
        self._tokenizer = self._get_tokenizer(tokenizer)
        self._label_resolver = self._get_label_resolver(tag_schema)

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
        source_dir_location = os.path.join(self._data_dir, 'croatian_ner',
                                           'CroatianNERDataset')

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
        if len(text) == 0:
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

    def _get_tokenizer(self, tokenizer_name):
        """
        Gets the tokenizer implementation associated with the given
        tokenizer name

        Parameters
        ----------
        tokenizer_name: str
            Name of the tokenizer

        Returns
        -------
        tokenizer: callable
            Tokenizer instance associated with the given tokenizer name
        """
        if tokenizer_name == 'split':
            return str.split

        raise ValueError('Unknown tokenizer specified: {}'
                         .format(tokenizer_name))

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

        raise ValueError('No label resolver for tag schema {} exists'
                         .format(tag_schema))

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
