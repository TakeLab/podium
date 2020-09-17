"""Module contains Croatian NER dataset."""
from podium.datasets.dataset import Dataset
from podium.storage.field import TokenizedField
from podium.storage import ExampleFactory
from podium.storage.vocab import Vocab
from podium.storage.resources.large_resource import LargeResource
from podium.dataload.ner_croatian import NERCroatianXMLLoader


class CroatianNERDataset(Dataset):
    """Croatian NER dataset.

    A single example in the dataset represents a single sentence in
    the input data.
    """

    def __init__(self, tokenized_documents, fields):
        """Dataset constructor.
        Users should use the static method get_dataset rather than invoking
        the constructor directly.

        Parameters
        ----------
        tokenized_documents : list(list(str, str))
            List of tokenized documents. Each document is represented
            as a list of tuples (token, label). The sentences in document are
            delimited by tuple (None, None)

        fields : list(Field)
            Dictionary that maps field name to the field
        """

        example_factory = ExampleFactory(fields)
        examples = []

        tokens = []
        labels = []

        for document in tokenized_documents:
            for line in document:
                if _is_delimiter_line(line):
                    examples.append(
                        example_factory.from_list((tokens, labels)))
                    tokens = []
                    labels = []
                else:
                    token, label = line
                    tokens.append(token)
                    labels.append(label)

        super().__init__(examples, fields)

    @classmethod
    def get_dataset(cls, tokenizer='split', tag_schema='IOB', fields=None, **kwargs):
        """Method downloads (if necessary) and loads the dataset.

        Parameters
        ----------
        tokenizer: str | callable
            Word-level tokenizer used to tokenize the input text

        tag_schema: str
            Tag schema used for constructing the token labels

            supported tag schemas:
                'IOB': the label of the beginning token of the entity is
                prefixed with 'B-', the remaining tokens that belong to the
                same entity are prefixed with 'I-'. The tokens that don't
                belong to any named entity are labeled 'O'

        fields : dict(str, Field)
            dictionary mapping field names to fields. If set to None, the
            default fields are used.

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


        Returns
        -------
        CroatianNERDataset
            The loaded dataset.
        """
        if not fields:
            fields = CroatianNERDataset.get_default_fields()

        path = LargeResource.BASE_RESOURCE_DIR

        ner_croatian_xml_loader = NERCroatianXMLLoader(path, tokenizer,
                                                       tag_schema, **kwargs)
        tokenized_documents = ner_croatian_xml_loader.load_dataset()

        ner_dataset = cls(tokenized_documents, fields.values())
        ner_dataset.finalize_fields()

        return ner_dataset

    @staticmethod
    def get_default_fields():
        """
        Method returns default Croatian NER dataset fields.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """

        tokens = TokenizedField(name="tokens", vocab=Vocab())
        labels = TokenizedField(name="labels", is_target=True)

        fields = {"tokens": tokens, "labels": labels}
        return fields


def _is_delimiter_line(line):
    """Checks if the line is delimiter line. Delimiter line is a tuple with
    all elements set to None.

    Parameters
    ----------
    line : tuple
        tuple representing line elements.

    Returns
    -------
        True if the line is delimiter line.
    """
    return not any(line)
