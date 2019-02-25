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
    def get_dataset(fields=None):
        """Downloads (if necessary) and loads the dataset. Not supported yet.
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

        LargeResource(**{
            LargeResource.RESOURCE_NAME: CatacxCommentsDataset.NAME,
            LargeResource.ARCHIVE: "zip",
            LargeResource.URI: CatacxCommentsDataset.URL
        })

        filepath = os.path.join(
            LargeResource.BASE_RESOURCE_DIR,
            CatacxCommentsDataset.DATASET_DIR,
            CatacxCommentsDataset.DATASET_FILE_NAME)

        return CatacxDataset.load_from_file(filepath, fields=fields)

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
    def get_sentences_tokenizer(key):
        def extractor_tokenizer(raw):
            tokens = []
            for sentence in raw:
                for token in sentence:
                    tokens.append(token.get(key))

            return tokens

        return extractor_tokenizer

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

        id_field = Field("id",
                         store_as_raw=True,
                         tokenize=False)

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

        pos_tag_field = Field("pos_tags",
                              store_as_raw=False,
                              tokenizer=CatacxDataset.get_sentences_tokenizer("pos_tag"))

        lemma_field = Field("lemmas",
                            store_as_raw=False,
                            tokenizer=CatacxDataset.get_sentences_tokenizer("lemma"))

        parent_ids_field = Field("parent_ids",
                                 store_as_raw=False,
                                 tokenizer=
                                 CatacxDataset.get_sentences_tokenizer("parent_id"))

        tokens_field = Field("tokens",
                             store_as_raw=False,
                             tokenizer=CatacxDataset.get_sentences_tokenizer("token"))

        dependency_tags_field = Field("dependency_tags",
                                      store_as_raw=False,
                                      tokenizer=CatacxDataset.get_sentences_tokenizer(
                                          "dependency_tag"))

        token_id_field = Field("id_tags",
                               store_as_raw=False,
                               tokenizer=CatacxDataset.get_sentences_tokenizer("id"))

        return {
            "sentiment": sentiment_field,
            "likes_cnt": likes_cnt_field,
            "message": message_field,
            "spam": spam_field,
            "emotions": emotions_field,
            "irony": irony_field,
            "speech_acts": speech_acts_field,
            "topics": topics_field,
            "cs": cs_field,
            "sentences": (pos_tag_field, lemma_field, parent_ids_field, tokens_field,
                          dependency_tags_field, token_id_field)
        }
