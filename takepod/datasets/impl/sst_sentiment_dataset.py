import os
from takepod.datasets.dataset import Dataset
from takepod.storage.field import Field, LabelField
from takepod.storage.example_factory import ExampleFactory, set_example_attributes
from takepod.storage.vocab import Vocab
from takepod.storage.resources.large_resource import LargeResource


class SST(Dataset):
    """The Stanford sentiment treebank dataset.

    Attributes
    ----------
    NAME : str
        dataset name
    URL : str
        url to the SST dataset
    DATASET_DIR : str
        name of the folder in the dataset containing train and test directories
    ARCHIVE_TYPE : str
        string that defines archive type, used for unpacking dataset
    TEXT_FIELD_NAME : str
        name of the field containing comment text
    LABEL_FIELD_NAME : str
        name of the field containing label value
    POSITIVE_LABEL : int
        positive sentiment label
    NEGATIVE_LABEL : int
        negative sentiment label
    """

    NAME = "sst"
    URL = 'https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    DATASET_DIR = "trees"
    ARCHIVE_TYPE = "zip"

    TRAIN_FILE = 'train.txt'
    VALID_FILE = 'dev.txt'
    TEST_FILE = 'test.txt'

    TEXT_FIELD_NAME = "text"
    LABEL_FIELD_NAME = "label"

    def __init__(self, file_path, fields, fine_grained=False, subtrees=False):
        """
        Dataset constructor. User should use static method
        get_dataset_splits rather than using the constructor directly.

        Parameters
        ----------
        dir_path : str
            path to the directory containing datasets

        fields : dict(str, Field)
            dictionary that maps field name to the field
        """
        LargeResource(**{
            LargeResource.RESOURCE_NAME: SST.NAME,
            LargeResource.ARCHIVE: SST.ARCHIVE_TYPE,
            LargeResource.URI: SST.URL})
        examples = self._create_examples(file_path=file_path, fields=fields,
                                         fine_grained=fine_grained, subtrees=subtrees)
        super(SST, self).__init__(
            **{"examples": examples, "fields": fields})

    @staticmethod
    def _create_examples(file_path, fields, fine_grained, subtrees):
        """
        Method creates examples for the sst dataset. Examples are arranged in two
        folders, one for examples with positive sentiment and other with negative
        sentiment. One file in each folder represents one example.

        Parameters
        ----------
        file_path : str
            file where examples for this split are stored
        fields : dict(str, Field)
            dictionary mapping field names to fields

        Returns
        -------
        examples : list(Example)
            list of examples from given dir_path
        """
        # Convert fields to list as the output is going to be a list
        fields_as_list = [fields[SST.TEXT_FIELD_NAME], fields[SST.LABEL_FIELD_NAME]]
        example_factory = ExampleFactory(fields_as_list)

        label_to_string_map = _label_to_string_map(fine_grained=fine_grained)
        def label_transform(label): return label_to_string_map[label]

        examples = []
        with open(file=file_path, mode='r', encoding='utf8') as fpr:
            for line in fpr:

                example = example_factory.from_fields_tree(line, subtrees=subtrees,
                                                           label_transform=label_transform)
                examples.append(example)
        return examples

    @staticmethod
    def get_dataset_splits(fields=None, fine_grained=False, subtrees=False):
        """Method loads and creates dataset splits for the SST dataset.

        Parameters
        ----------
        fields : dict(str, Field), optional
            dictionary mapping field name to field, if not given method will
            use ```get_default_fields```. User should use default field names
            defined in class attributes.

        Returns
        -------
        (train_dataset, valid_dataset, test_dataset) : (Dataset, Dataset, Dataset)
            tuple containing train, valid and test dataset
        """
        data_location = os.path.join(LargeResource.BASE_RESOURCE_DIR,
                                     SST.DATASET_DIR)
        if fields is None:
            fields = SST.get_default_fields()

        train_dataset = SST(
            file_path=os.path.join(
                data_location, SST.TRAIN_FILE),
            fields=fields, fine_grained=fine_grained, subtrees=subtrees)

        valid_dataset = SST(
            file_path=os.path.join(
                data_location, SST.VALID_FILE),
            fields=fields, fine_grained=fine_grained, subtrees=subtrees)

        test_dataset = SST(
            file_path=os.path.join(
                data_location, SST.TEST_FILE),
            fields=fields, fine_grained=fine_grained, subtrees=subtrees)

        # If not fine-grained, return binary task: filter out neutral instances
        if not fine_grained:
            def filter_neutral(example):
                return example.label[0] != 'neutral'
            train_dataset.filter(predicate=filter_neutral, inplace=True)
            valid_dataset.filter(predicate=filter_neutral, inplace=True)
            test_dataset.filter(predicate=filter_neutral, inplace=True)

        train_dataset.finalize_fields()
        return (train_dataset, valid_dataset, test_dataset)

    @staticmethod
    def get_default_fields():
        """Method returns default Imdb fields: text and label.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """
        text = Field(name=SST.TEXT_FIELD_NAME, vocab=Vocab(),
                     tokenizer='split', language="en", tokenize=True,
                     store_as_raw=False, eager=False)
        label = LabelField(name=SST.LABEL_FIELD_NAME, eager=False)
        return {SST.TEXT_FIELD_NAME: text,
                SST.LABEL_FIELD_NAME: label}


def _label_to_string_map(fine_grained):
    pre = 'very ' if fine_grained else ''
    return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
            '3': 'positive', '4': pre + 'positive'}

def _get_label_str(label, fine_grained):
    return _label_to_string_map(fine_grained)[label]
