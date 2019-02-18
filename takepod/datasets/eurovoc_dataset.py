"Module contains EuroVoc dataset."
import json
from enum import Enum
from takepod.storage import dataset
from takepod.storage.example import Example
from takepod.storage.field import Field, TokenizedField
from takepod.storage.vocab import Vocab


class LabelRank(Enum):
    """Levels of labels in EuroVoc.
    """
    THEZAURUS = "thezaurus"
    MICRO_THEZAURUS = "micro_thezaurus"
    TERM = "term"


class Label():
    """Label in EuroVoc dataset.

    Labels are assigned to documents. One document has multiple labels.
    Labels have a hierarchy in which one label can have one or more parents (broader
    terms). All labels apart from thezaurus rank labels have at least one parent.
    Apart from parents, labels can also have similar labels which describe related
    areas, but aren't connected by the label hierarchy.
    """

    def __init__(self, name, id, parents, similar_terms, rank, thezaurus=None,
                 micro_thezaurus=None):
        """Defines a single label in the EuroVoc dataset.

        Parameters
        ----------
        name : str
            name of the label
        id : int
            numerical id of the label
        parents : list(int)
            list of ids of direct parents
        similar_terms : list(int)
            list of ids of similar terms
        rank : LabelRank
            rank of the label
        thezaurus : int
            id of the thezaurus of the label (if the label represents a
            thezaurus, it has its own id listed in this field)
        micro_thezaurus : int
            id of the microthezaurus of the label (if the label represents
            a microthezaurus, it has its own id listed in this field)
        """
        self.name = name
        self.id = id
        self.parents = parents
        self.similar_terms = similar_terms
        self.rank = rank
        self.micro_thezaurus = micro_thezaurus
        self.thezaurus = thezaurus


class EuroVocDataset(dataset.Dataset):
    """EuroVoc dataset class that contains labeled documents and the label hierarchy.
    """

    def __init__(self, dataset_path, label_hierarchy_path, fields=None):
        """Dataset constructor.

        Parameters
        ----------
        dataset_path : str
            path to json file containing the dataset

        label_hierarchy_path : str
            path to json file containing label hierarchy

        fields : dict(str, Field)
            dictionary that maps field name to the field
        """
        self._label_hierarchy = EuroVocDataset._load_label_hierarchy(label_hierarchy_path)
        if not fields:
            fields = EuroVocDataset._get_default_fields()
        unpacked_fields = dataset.unpack_fields(fields=fields)
        examples = EuroVocDataset._create_examples(dataset_path, fields)
        super(EuroVocDataset, self).__init__(
            **{"examples": examples, "fields": unpacked_fields})

    def get_label_hierarchy(self):
        """Returns the label hierarchy.

        Returns
        -------
        label_hierarchy : dict(int, Label)
            dictionary that maps label id to label
        """
        return self._label_hierarchy

    @staticmethod
    def _load_label_hierarchy(label_hierarchy_path):
        """Loads the EuroVoc label hierarchy from a json file.

        Returns
        -------
        label_hierarchy : dict(int, Label)
            dictionary that maps label id to label
        """

        labels = {}
        with open(label_hierarchy_path, "r", encoding="utf-8") as input_file:
            data = json.load(input_file)
            for key in data:
                entry = data[key]
                label = Label(name=entry['name'], id=entry['id'],
                              parents=entry['parents'],
                              similar_terms=entry['similar_terms'], rank=entry['rank'],
                              thezaurus=entry['thezaurus'],
                              micro_thezaurus=entry['micro_thezaurus'])
                labels[label.id] = label
        return labels

    @staticmethod
    def _create_examples(path, fields):
        """Method creates examples for EuroVoc dataset. All examples are stored in one
        json file.

        Parameters
        ----------
        path : str
            json file where examples are stored
        fields : dict(str, Field)
            dictionary mapping field names to fields

        Returns
        -------
        examples : list(Example)
            list of examples from given file
        """
        examples = []
        with open(path, "r", encoding="utf-8") as input_file:
            examples_dict = json.load(input_file)
            for key in examples_dict:
                examples.append(Example.fromdict(data=examples_dict[key],
                                                 fields=fields))
        return examples

    @staticmethod
    def _get_default_fields():
        """Method returns default EuroVoc fields: title, text and labels.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """
        title = Field(name="title", vocab=Vocab(), tokenizer='split', language="hr",
                      sequential=True, store_raw=False)
        text = Field(name="text", vocab=Vocab(), tokenizer='split', language="hr",
                     sequential=True, store_raw=False)
        labels = TokenizedField(name="labels", vocab=Vocab(specials=()),
                                is_target=True)
        fields = {"title": title, "text": text, "labels": labels}
        return fields
