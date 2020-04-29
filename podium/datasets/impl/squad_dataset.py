"""
Module contains Stanford Question Answering Dataset (SQuAD 2.0).
Dataset webpage: https://rajpurkar.github.io/SQuAD-explorer/
"""

import json
import os
from podium.datasets import Dataset
from podium.storage import Vocab, Field, LargeResource, ExampleFactory


class SQuADDataset(Dataset):
    """
    SQuAD dataset class

    Attributes
    ----------
    TRAIN_FILE_NAME : str
        name of the file containing train dataset
    DEV_FILE_NAME : str
        name of the file containing dev dataset
    URL_TRAIN : str
        url to the train SQuAD dataset
    URL_DEV : str
        url to the dev SQuAD dataset
    COMPRESSION_TYPE : str
        string that defines compression type, used for unpacking dataset

    QUESTION_FIELD_NAME : str
        name of the field containing question
    ANSWER_FIELD_NAME : str
        name of the field containing answer (if question is not impossible)
    PLAUSIBLE_ANSWER_FIELD_NAME : str
        name of the field containing plausible answer (if question is impossible)
    CONTEXT_FIELD_NAME : str
        name of the field containing question context
    IS_IMPOSSIBLE_FIELD_NAME : str
        name of the field containing question is impossible flag
    """

    TRAIN_FILE_NAME = "SQuAD_train.json"
    DEV_FILE_NAME = "SQuAD_dev.json"
    URL_TRAIN = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    URL_DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    COMPRESSION_TYPE = "gz"

    QUESTION_FIELD_NAME = "question"
    ANSWER_FIELD_NAME = "answer"
    PLAUSIBLE_ANSWER_FIELD_NAME = "plausible_answer"
    CONTEXT_FIELD_NAME = "context"
    IS_IMPOSSIBLE_FIELD_NAME = "is_impossible"

    def __init__(self, file_path, fields):
        """
        Dataset constructor. User should use static method
        get_train_dev_dataset rather than using constructor directly.

        ----------
        file_path : str
            path to the dataset file

        fields : dict(str, Field)
            dictionary that maps field name to the field
        """

        LargeResource(**{
            LargeResource.RESOURCE_NAME: SQuADDataset.TRAIN_FILE_NAME,
            LargeResource.COMPRESSION: SQuADDataset.COMPRESSION_TYPE,
            LargeResource.URI: SQuADDataset.URL_TRAIN
        })

        LargeResource(**{
            LargeResource.RESOURCE_NAME: SQuADDataset.DEV_FILE_NAME,
            LargeResource.COMPRESSION: SQuADDataset.COMPRESSION_TYPE,
            LargeResource.URI: SQuADDataset.URL_DEV
        })

        examples = self._create_examples(file_path, fields)
        super(SQuADDataset, self).__init__(
            **{"examples": examples, "fields": fields})

    @staticmethod
    def _create_examples(file_path, fields):
        """
        Method creates examples for SQuAD dataset, from a single input `.json` file.

        Parameters
        ----------
        file_path : str
            path to the `.json` file in which examples are stored
        fields : dict(str, Field)
            dictionary mapping field names to fields

        Returns
        -------
        examples : list(Example)
            list of examples from given file_path
        """

        example_factory = ExampleFactory(fields)
        examples = []

        with open(file=file_path, mode='r', encoding='utf8') as in_file:
            dataset = json.load(in_file)
            for article in dataset['data']:
                for paragraph in article['paragraphs']:
                    for question_answer in paragraph['qas']:
                        if 'plausible_answers' in question_answer:
                            for plausible_answer in question_answer['plausible_answers']:
                                examples.append(example_factory.from_dict({
                                    SQuADDataset.QUESTION_FIELD_NAME: question_answer[
                                        'question'],
                                    SQuADDataset.ANSWER_FIELD_NAME: '',
                                    SQuADDataset.PLAUSIBLE_ANSWER_FIELD_NAME:
                                        plausible_answer['text'],
                                    SQuADDataset.CONTEXT_FIELD_NAME: paragraph['context'],
                                    SQuADDataset.IS_IMPOSSIBLE_FIELD_NAME:
                                        question_answer['is_impossible']
                                }))
                        else:
                            for answer in question_answer['answers']:
                                examples.append(example_factory.from_dict({
                                    SQuADDataset.QUESTION_FIELD_NAME: question_answer[
                                        'question'],
                                    SQuADDataset.ANSWER_FIELD_NAME: answer['text'],
                                    SQuADDataset.PLAUSIBLE_ANSWER_FIELD_NAME: '',
                                    SQuADDataset.CONTEXT_FIELD_NAME: paragraph['context'],
                                    SQuADDataset.IS_IMPOSSIBLE_FIELD_NAME:
                                        question_answer[
                                            'is_impossible']
                                }))

        return examples

    @staticmethod
    def get_train_dev_dataset(fields=None):
        """
        Method creates train and dev dataset for SQuAD dataset..

        Parameters
        ----------
        fields : dict(str, Field), optional
            dictionary mapping field name to field, if not given method will
            use ```get_default_fields```.

        Returns
        -------
        (train_dataset, dev_dataset) : (Dataset, Dataset)
            tuple containing train and dev datasets
        """
        train_file_path = os.path.join(LargeResource.BASE_RESOURCE_DIR,
                                       SQuADDataset.TRAIN_FILE_NAME)

        dev_file_path = os.path.join(LargeResource.BASE_RESOURCE_DIR,
                                     SQuADDataset.DEV_FILE_NAME)

        if not fields:
            fields = SQuADDataset.get_default_fields()

        train_dataset = SQuADDataset(file_path=train_file_path, fields=fields)
        dev_dataset = SQuADDataset(file_path=dev_file_path, fields=fields)

        train_dataset.finalize_fields()
        return (train_dataset, dev_dataset)

    @staticmethod
    def get_default_fields():
        """
        Method returns default SQuAD dataset fields:
        question, answer, plausible_answer context and is_impossible.

        Returns
        -------
        fields : dict(str, Field)
            dictionary mapping field name to field
        """
        question = Field(name=SQuADDataset.QUESTION_FIELD_NAME, vocab=Vocab(),
                         tokenizer='split', tokenize=True,
                         store_as_raw=False)
        answer = Field(name=SQuADDataset.ANSWER_FIELD_NAME, vocab=Vocab(),
                       tokenizer='split', tokenize=True,
                       store_as_raw=False,
                       is_target=True)
        plausible_answer = Field(name=SQuADDataset.PLAUSIBLE_ANSWER_FIELD_NAME,
                                 vocab=Vocab(), tokenizer='split',
                                 tokenize=True,
                                 store_as_raw=False,
                                 is_target=True)
        context = Field(name=SQuADDataset.CONTEXT_FIELD_NAME, vocab=Vocab(),
                        tokenizer='split', tokenize=True,
                        store_as_raw=False)
        is_impossible = Field(name=SQuADDataset.IS_IMPOSSIBLE_FIELD_NAME,
                              vocab=Vocab(specials=()), tokenize=False,
                              store_as_raw=True)
        fields = {SQuADDataset.QUESTION_FIELD_NAME: question,
                  SQuADDataset.ANSWER_FIELD_NAME: answer,
                  SQuADDataset.PLAUSIBLE_ANSWER_FIELD_NAME: plausible_answer,
                  SQuADDataset.CONTEXT_FIELD_NAME: context,
                  SQuADDataset.IS_IMPOSSIBLE_FIELD_NAME: is_impossible}

        return fields
