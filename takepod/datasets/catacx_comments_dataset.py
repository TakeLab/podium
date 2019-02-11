"""Module contains the catacx dataset."""
import json
import os
from takepod.storage.large_resource import LargeResource
from takepod.storage import dataset
from takepod.storage.example import Example
from takepod.storage.field import Field


class CatacxCommentsDataset(dataset.Dataset):
    """Simple Catacx dataset. Contains only the comments."""

    NAME = "CatacxCommentsDataset"
    DATASET_FILE_NAME = "catacx_dataset.json"
    DATASET_DIR = os.path.join("Catacx", NAME)
    URL = None  # TODO Add real URL

    def __init__(self, dir_path, fields=None):
        """Dataset constructor, should be given the path to the .json file which contains the Catacx dataset.

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
    def get_dataset(fields=None):
        """Downloads (if necessary) and loads the dataset. Not fully implemented yet.

        :return:
            the loaded Catacx comment dataset
        """

        raise NotImplementedError("Downloading is not implemented yet")

        LargeResource(**{
            LargeResource.RESOURCE_NAME: CatacxCommentsDataset.NAME,
            LargeResource.ARCHIVE: "zip",
            LargeResource.URI: CatacxCommentsDataset.URL
        })

        filepath = os.path.join(
            LargeResource.BASE_RESOURCE_DIR
            , CatacxCommentsDataset.DATASET_DIR
            , CatacxCommentsDataset.DATASET_FILE_NAME)

        return CatacxCommentsDataset(filepath, fields=fields)

    @staticmethod
    def _get_comments(ds):
        """
        :param ds:
            Loaded Catacx dataset in dict form.

        :return:
            Generator iterating trough comments in the dataset.
        """
        for post in ds:
            for comment in post["comments"]:
                yield comment

    @staticmethod
    def _create_examples(dir_path, fields):
        """
        Loads examples from the dataset file.

        :param dir_path: str
            File path to the dataset .json file.

        :param fields:
            Dict of fields to be loaded.
        :return:
            List of Examples loaded from the dataset file.
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
        Method returns a dict of default Catacx comment fields.
        fields : likes_cnt, id, likes_cnt, message


        Returns
        -------
        fields : dict(str, Field)
            dict containing all default Catacx fields
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
            "author_id": author_id,
            "id": id,
            "likes_cnt": likes_cnt,
            "message": message
        }
