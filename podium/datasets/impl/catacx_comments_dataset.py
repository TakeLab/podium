"""
Module contains the catacx dataset.
"""
import json
import os

from podium.datasets.dataset import Dataset
from podium.datasets.example_factory import ExampleFactory
from podium.field import Field
from podium.storage.resources.large_resource import LargeResource


class CatacxCommentsDataset(Dataset):
    """
    Simple Catacx dataset.

    Contains only the comments.
    """

    NAME = "CatacxCommentsDataset"
    DATASET_FILE_NAME = "catacx_dataset.json"
    DATASET_DIR = os.path.join("Catacx", NAME)
    URL = None  # TODO Add real URL

    def __init__(self, dir_path, fields=None):
        """
        Dataset constructor, should be given the path to the .json file which
        contains the Catacx dataset.

        Parameters
        ----------
        dir_path : str
            path to the file containing the dataset.

        fields : dict(str, Field)
            dictionary that maps field name to the field
            if passed None the default set of fields will be used
        """
        fields = fields if fields else CatacxCommentsDataset.get_default_fields()
        examples = CatacxCommentsDataset._create_examples(dir_path, fields)
        super(CatacxCommentsDataset, self).__init__(examples, fields)

    @staticmethod
    def get_dataset(fields=None):
        """
        Downloads (if necessary) and loads the dataset. Not supported yet.
        Raises NotImplementedError if called.

        Parameters
        ----------
        fields : dict(str, Field)
            dictionary that maps field name to the field
            if passed None the default set of fields will be used.

        Returns
        -------
        CatacxCommentsDataset
            The loaded dataset.
        """

        raise NotImplementedError("Downloading is not implemented yet")

        LargeResource(
            **{
                LargeResource.RESOURCE_NAME: CatacxCommentsDataset.NAME,
                LargeResource.ARCHIVE: "zip",
                LargeResource.URI: CatacxCommentsDataset.URL,
            }
        )

        filepath = os.path.join(
            LargeResource.BASE_RESOURCE_DIR,
            CatacxCommentsDataset.DATASET_DIR,
            CatacxCommentsDataset.DATASET_FILE_NAME,
        )

        return CatacxCommentsDataset(filepath, fields=fields)

    @staticmethod
    def _get_comments(ds):
        """
        Generator iterating trough the comments in the Catacx dataset.

        Parameters
        ----------
        ds
            Dataset loaded from a .json file in dict form.

        Yields
        ------
            The next comment in the dataset.
        """

        for post in ds:
            for comment in post["comments"]:
                yield comment

    @staticmethod
    def _create_examples(dir_path, fields):
        """
        Loads the dataset and extracts Examples.

        Parameters
        ----------
        dir_path : str
            Path to the .json file wich contains the dataset.

        fields : dict(str, Field)
            dictionary that maps field name to the field.

        Returns
        -------
        list(Example)
            A list of examples containing comments from the Catacx dataset.
        """
        example_factory = ExampleFactory(fields)
        with open(os.path.expanduser(dir_path), encoding="utf8") as f:
            ds = json.load(f)

        examples = []

        for comment in CatacxCommentsDataset._get_comments(ds):
            examples.append(example_factory.from_dict(comment))

        return examples

    @staticmethod
    def get_default_fields():
        """Method returns a dict of default Catacx comment fields.
        fields : author_name, author_id, id, likes_cnt, message


        Returns
        -------
        fields : dict(str, Field)
            dict containing all default Catacx fields
        """
        # TODO: Add remaining fields when NestedFields is implemented
        # Fields not yet supported or not important:
        #
        # replies - List of replies
        # smileys - list of Smileys
        # likes - list of Likes
        # sentences - list of Sentences preprocessed message of the comment
        # created_time - JSON date
        # cs

        author_name = Field(name="author_name", tokenizer=None, keep_raw=True)

        id = Field(name="id", tokenizer=None, keep_raw=True)

        likes_cnt = Field(name="likes_cnt", tokenizer=None, numericalizer=int)

        message = Field(name="message", tokenizer="split")

        author_id = Field(name="author_id", tokenizer=None, keep_raw=True)

        return {
            "author_name": author_name,
            "author_id": author_id,
            "id": id,
            "likes_cnt": likes_cnt,
            "message": message,
        }
