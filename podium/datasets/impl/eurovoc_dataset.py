"""Module contains EuroVoc dataset."""
import os
import re
import functools
import logging
from podium.datasets.dataset import Dataset
from podium.storage.example_factory import ExampleFactory, set_example_attributes
from podium.storage import Field, MultilabelField
from podium.storage import Vocab
from podium.preproc.stop_words import CROATIAN_EXTENDED
from podium.preproc.lemmatizer.croatian_lemmatizer import get_croatian_lemmatizer_hook

_LOGGER = logging.getLogger(__name__)


class EuroVocDataset(Dataset):
    """EuroVoc dataset class that contains labeled documents and the label hierarchy.
    """

    def __init__(self,
                 eurovoc_labels,
                 crovoc_labels,
                 documents,
                 mappings,
                 fields=None):
        """Dataset constructor.

        Parameters
        ----------
        eurovoc_labels : dict(int : Label)
            dictionary mapping eurovoc label_ids to labels

        crovoc_labels : dict(int : Label)
            dictionary mapping crovoc label_ids to labels

        documents : list(Document)
            list of all documents in dataset

        mappings : dict(int : list(int))
            dictionary that maps documents_ids to list of their label_ids

        fields : dict(str : Field)
            dictionary that maps field name to the field
        """
        self._eurovoc_label_hierarchy = eurovoc_labels
        self._crovoc_label_hierarchy = crovoc_labels

        if not fields:
            fields = EuroVocDataset.get_default_fields()

        examples = EuroVocDataset._create_examples(
            fields=fields,
            documents=documents,
            mappings=mappings,
            eurovoc_label_hierarchy=eurovoc_labels,
            crovoc_label_hierarchy=crovoc_labels)
        super(EuroVocDataset, self).__init__(
            **{"examples": examples, "fields": fields})

    def get_eurovoc_label_hierarchy(self):
        """Returns the EuroVoc label hierarchy.

        Returns
        -------
        dict(int : Label)
            dictionary that maps label id to label
        """
        return self._eurovoc_label_hierarchy

    def get_crovoc_label_hierarchy(self):
        """Returns CroVoc label hierarchy.

        Returns
        -------
        dict(int : Label)
            dictionary that maps label id to label
        """
        return self._crovoc_label_hierarchy

    @staticmethod
    def _create_examples(fields, documents, mappings, eurovoc_label_hierarchy,
                         crovoc_label_hierarchy):
        """Method creates examples for EuroVoc dataset.

        Examples are created from the given documents and mappings. Documents that don't
        have a matching entry in the mappings are ignored. Mappings that don't match to
        any document are ignored as well. Furthermore, if a document is labeled with a
        label that is not present in eurovoc nor crovoc label hierarchy, the label is
        ignored.

        Parameters
        ----------
        fields : dict(str : Field)
            dictionary mapping field names to fields

        documents : list(Document)
            list of all documents in dataset

        mappings : dict(int : list(int))
            dictionary that maps documents_ids to list of their label_ids

         eurovoc_labels_hierarchy : dict(int : Label)
            dictionary mapping eurovoc label_ids to labels

        crovoc_label_hierarchy : dict(int : Label)
            dictionary mapping crovoc label_ids to labels

        Returns
        -------
        list(Example)
            list of examples created from the given documents and mappings
        """
        example_factory = ExampleFactory(fields)
        examples = []

        for document in documents:
            # document filename format is NNXXXXX.xml, where XXXXX is document_id
            document_id = int(os.path.splitext(document.filename)[0].replace("NN", ""))
            if document_id not in mappings:
                debug_msg = "Document {} not found in mappings".format(document_id)
                _LOGGER.debug(debug_msg)
                continue

            labels = mappings[document_id]
            eurovoc_labels = []
            crovoc_labels = []
            for label in labels:
                if label in eurovoc_label_hierarchy:
                    eurovoc_labels.append(label)
                elif label in crovoc_label_hierarchy:
                    crovoc_labels.append(label)
                else:
                    debug_msg = "Document {} has label {} which is not present in the"\
                                "given label hierarchies.".format(document_id, label)
                    _LOGGER.debug(debug_msg)

            example = example_factory.create_empty_example()
            set_example_attributes(example, fields["title"], document.title)
            set_example_attributes(example, fields["text"], document.text)
            set_example_attributes(example, fields["eurovoc_labels"],
                                   eurovoc_labels)
            set_example_attributes(example, fields["crovoc_labels"], crovoc_labels)
            examples.append(example)

        return examples

    def is_ancestor(self, label_id, example):
        """Checks if the given label_id is an ancestor of any labels of the example.

        Parameters
        ----------
        label_id : int
            id of the label

        example : Example
            example from dataset

        Returns
        -------
        boolean:
            True if label is ancestor to any of the example labels, False otherwise
        """
        # the given label can be either in crovoc or in eurovoc label hierarchy, therefore
        # we need to check both hierarchies for ancestors
        example_labels = example.eurovoc_labels[1]
        example_labels.extend(example.crovoc_labels[1])

        for example_label_id in example_labels:
            if example_label_id in self._eurovoc_label_hierarchy:
                example_label = self._eurovoc_label_hierarchy[example_label_id]
            elif example_label_id in self._crovoc_label_hierarchy:
                example_label = self._crovoc_label_hierarchy[example_label_id]

            if label_id in example_label.all_ancestors:
                return True
        return False

    def get_direct_parents(self, label_id):
        """Returns ids of direct parents of the label with the given label id.

        Parameters
        ----------
        label_id : int
            id of the label

        Returns
        -------
        list(int)
            list of label_ids of all direct parents of the given label or None if the
            label is not present in the dataset label hierarchies

        """
        if label_id in self._eurovoc_label_hierarchy:
            label = self._eurovoc_label_hierarchy[label_id]
            return label.direct_parents

        elif label_id in self._crovoc_label_hierarchy:
            label = self._crovoc_label_hierarchy[label_id]
            return label.direct_parents

        return None

    def get_all_ancestors(self, label_id):
        """Returns ids of all ancestors of the label with the given label id.

        Parameters
        ----------
        label_id : int
            id of the label

        Returns
        -------
        list(int)
            list of label_ids of all ancestors of the given label or None if the
            label is not present in the dataset label hierarchies

        """
        if label_id in self._eurovoc_label_hierarchy:
            label = self._eurovoc_label_hierarchy[label_id]
            return label.all_ancestors

        elif label_id in self._crovoc_label_hierarchy:
            label = self._crovoc_label_hierarchy[label_id]
            return label.all_ancestors

        return None

    @staticmethod
    def get_default_fields():
        """Method returns default EuroVoc fields: title, text, eurovoc and crovoc labels.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """
        title = Field(name="title", vocab=Vocab(), tokenizer='split', language="hr",
                      tokenize=True, store_as_raw=False)
        text = Field(name="text", vocab=Vocab(keep_freqs=True),
                     tokenizer='split', tokenize=True, store_as_raw=False)
        text.add_posttokenize_hook(functools.partial(remove_nonalpha_and_stopwords,
                                                     stop_words=set(CROATIAN_EXTENDED)))
        text.add_posttokenize_hook(get_croatian_lemmatizer_hook())
        labels = MultilabelField(name="eurovoc_labels", vocab=Vocab(specials=()))
        crovoc_labels = MultilabelField(name="crovoc_labels", vocab=Vocab(specials=()))
        fields = {"title": title, "text": text, "eurovoc_labels": labels,
                  "crovoc_labels": crovoc_labels}
        return fields


def remove_nonalpha_and_stopwords(raw, tokenized, stop_words):
    """Removes all non alphabetical characters and stop words from tokens.

    Parameters
    ----------
    raw : string
        raw text

    tokenized : list(str)
        tokenized text

    Returns
    -------
    tuple(str, list(str))
    """
    tokens = []
    pattern = re.compile(r'[\W_]+')
    for token in tokenized:
        pattern.sub('', token)
        if len(token) > 1 and token not in stop_words:
            tokens.append(token)
    return (raw, tokens)
