"""Module contains the catacx dataset."""
import json
from takepod.storage import dataset
from takepod.storage.example import Example
from takepod.storage.field import Field
from takepod.storage.vocab import Vocab


class CatacxCommentsDataset(dataset.Dataset):
    """Simple catacx dataset. Contains only the comments."""

    NAME = "catacx"

    def __init__(self, dir_path, fields=None):
        """Dataset constructor, should be given the path to the .json file which contains the catacx dataset.

        ATTRIBUTES
        ----------
        dir_path : str
            path to the file containing the dataset

        fields : dict(str, Field)
            dictionary that maps field name to the field
            if passed None the default set of fields will be used
        """
        fields = fields if fields else CatacxCommentsDataset._get_default_fields()
        examples = CatacxCommentsDataset._create_examples(dir_path, fields)
        unpacked_fields = dataset.unpack_fields(fields)
        super(CatacxCommentsDataset, self).__init__(examples, unpacked_fields)

    @staticmethod
    def _get_comments(ds):
        for post in ds:
            for comment in post["comments"]:
                yield comment

    @staticmethod
    def _create_examples(dir_path, fields):
        """
        Loads examples from the dataset file.

        :param dir_path: str

        :param fields:
        :return:
        """
        with open(dir_path, encoding="utf8", mode="r") as f:
            ds = json.load(f)

        examples = list()

        for comment in CatacxCommentsDataset._get_comments(ds):
            examples.append(Example.fromdict(comment, fields))

        return examples

    @staticmethod
    def _get_default_fields():
        """
        Method returns a dict of default catacx comment fields.
        fields : likes_cnt, id, likes_cnt, message


        Returns
        -------
        fields : dict(str, Field)
            dict containing all default catacx fields
        """
        # TODO: Add remaining fields when NestedFields is implemented
        # commented lines are fields not yet supported or not important
        # listed in the order they appear in the JSON of the comment

        # replies - List of replies

        author_name = Field(name='author_name', sequential=False)

        id = Field(name='id', sequential=False)

        likes_cnt = Field(name="likes_cnt", vocab=None,
                          sequential=False,
                          custom_numericalize=int)

        # smileys - list of Smileys

        # likes - list of Likes

        # sentences - list of Sentences preprocessed message of the comment

        # created_time - JSON date

        message = Field(name='message', sequential=True, store_raw=False,
                        tokenizer='split', language='hr')

        author_id = Field(name='author_id', sequential=False)
        # cs - List of something, not documented in the official catacx documentation

        return {
            "author_name": author_name,
            "id": id,
            "likes_cnt": likes_cnt,
            "message": message
        }
