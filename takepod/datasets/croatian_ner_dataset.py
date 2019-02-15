"""Module contains Croatian NER dataset."""
from takepod.storage import dataset
from takepod.storage.field import TokenizedField
from takepod.storage.vocab import Vocab
from takepod.storage.large_resource import LargeResource
from takepod.dataload.ner_croatian import NERCroatianXMLLoader
from takepod.datasets.sequence_labelling_dataset import SequenceLabellingDataset


class CroatianNERDataset(SequenceLabellingDataset):
    """Croatian NER dataset.
    A single example represents a single sentence in the input data.

    Attributes
    ----------
    NAME : str
        Name of the dataset.
    """

    NAME = "CroatianNERDataset"

    @classmethod
    def get_dataset(cls, tokenizer='split', tag_schema='IOB', fields=None, **kwargs):
        """
        Method downloads (if necessary) and loads the dataset.

        Parameters
        ----------
        tokenizer: str | callable
            Word-level tokenizer used to tokenize the input text
        tag_schema: str
            Tag schema used for constructing the token labels
            - supported tag schemas:
                - 'IOB': the label of the beginning token of the entity is
                prefixed with 'B-', the remaining tokens that belong to the
                same entity are prefixed with 'I-'. The tokens that don't
                belong to any named entity are labeled 'O'
        fields : dict(str, Field)
            dictionary mapping field names to fields. If set to None, the
            default fields are used.

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

        Returns
        -------
        CroatianNERDataset
            The loaded dataset.
        """
        if not fields:
            fields = CroatianNERDataset._get_default_fields()

        path = LargeResource.BASE_RESOURCE_DIR

        ner_croatian_xml_loader = NERCroatianXMLLoader(path, tokenizer,
                                                       tag_schema, **kwargs)
        tokenized_documents = ner_croatian_xml_loader.load_dataset()

        ner_dataset = cls(tokenized_documents, dataset.unpack_fields(fields))
        ner_dataset.finalize_fields()

        return ner_dataset

    @staticmethod
    def _get_default_fields():
        """
        Method returns default Croatian NER dataset fields.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """

        tokens = TokenizedField(name="tokens", vocab=Vocab(specials=()))
        labels = TokenizedField(name="labels", is_target=True)

        fields = {"tokens": tokens, "labels": labels}
        return fields
