
import os

from takepod.storage import HierarchicalDataset, Example, Field, \
    MultilabelField, Vocab


class CatacxDataset(HierarchicalDataset):
    """Catacx dataset."""

    NAME = "CatacxDataset"
    DATASET_FILE_NAME = "catacx_dataset.json"
    DATASET_DIR = os.path.join("Catacx", NAME)
    URL = None  # TODO Add real URL

    def __init__(self, dir_path, fields=None):
        """Dataset constructor, should be given the path to the .json file which contains
        the Catacx dataset.

        Parameters
        ----------
        dir_path : str
            path to the file containing the dataset.

        fields : dict(str, Field)
            dictionary that maps field name to the field
            if passed None the default set of fields will be used
        """
        fields = fields if fields else CatacxDataset._get_default_fields()
        super().__init__(fields, CatacxDataset.get_catacx_parser())

    @staticmethod
    def load_from_file(path, fields=None):
        fields = fields if fields else CatacxDataset._get_default_fields()
        parser = CatacxDataset._get_catacx_parser()

        with open(path, mode='r') as f:
            ds_str = f.read()

        return HierarchicalDataset.from_json(ds_str, fields, parser)


    @staticmethod
    def _get_catacx_parser():
        def catacx_parser(raw, fields, depth):
            example = Example.fromdict(raw, fields)

            if depth == 0:
                children = raw.get('comments')

            elif depth == 1:
                children = raw.get('replies')

            else:
                children = []

            return example, children

        return catacx_parser


    @staticmethod
    def _get_default_fields():
        """
        Method returns a dict of default Catacx comment fields.
        fields : author_name, author_id, id, likes_cnt, message


        Returns
        -------
        fields : dict(str, Field)
            dict containing all default Catacx fields
        """
        # TODO: Add remaining fields when NestedFields is implemented
        # Fields not yet supported or not important:
        #
        # reactions
        # reaction_cnt
        # smileys
        # author_name
        # id
        # author_id

        # replies - List of replies
        # smileys - list of Smileys
        # likes - list of Likes
        # sentences - list of Sentences preprocessed message of the comment
        # created_time - JSON date
        # cs
        sentiment_field = Field("sentiment",
                                store_as_raw=True,
                                tokenize=False,
                                custom_numericalize=float,
                                default_value_callable=Field.empty_vector_callable())

        likes_cnt_field = Field("likes_cnt",
                                store_as_raw=True,
                                tokenize=False,
                                custom_numericalize=int,
                                default_value_callable=Field.empty_vector_callable())

        message_field = Field(name='message',
                              vocab=Vocab(),
                              tokenize=True,
                              store_as_raw=False,
                              tokenizer='split',
                              language='hr')

        spam_field = Field("spam",
                           store_as_raw=True,
                           tokenize=False,
                           custom_numericalize=int,
                           default_value_callable=Field.empty_vector_callable())

        emotions_field = MultilabelField("emotions",
                                         vocab=Vocab(specials=()),
                                         default_value_callable=
                                         Field.empty_vector_callable())

        irony_field = Field("irony",
                            store_as_raw=True,
                            tokenize=False,
                            custom_numericalize=int,
                            default_value_callable=Field.empty_vector_callable())

        speech_acts_field = MultilabelField("speech_acts",
                                            vocab=Vocab(specials=()),
                                            default_value_callable=
                                            Field.empty_vector_callable())

        topics_field = MultilabelField("topics",
                                       vocab=Vocab(specials=()),
                                       default_value_callable=
                                       Field.empty_vector_callable())

        cs_field = MultilabelField("cs",
                                   vocab=Vocab(specials=()),
                                   default_value_callable=Field.empty_vector_callable())

        return {
            "sentiment": sentiment_field,
            "likes_cnt": likes_cnt_field,
            "message": message_field,
            "spam": spam_field,
            "emotions": emotions_field,
            "irony": irony_field,
            "speech_acts": speech_acts_field,
            "topics": topics_field,
            "cs": cs_field
        }
