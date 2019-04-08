import os

from takepod.storage import HierarchicalDataset, Example, Field, \
    MultilabelField, Vocab, ExampleFactory


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
        fields = fields if fields else CatacxDataset.get_default_fields()
        super().__init__(fields, CatacxDataset._get_catacx_parser())

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
        CatacxDataset
            The loaded dataset.
        """

        raise NotImplementedError("Downloading is not implemented yet")
        # TODO: uncomment when the dataset can be downloaded
        # LargeResource(**{
        #     LargeResource.RESOURCE_NAME: CatacxDataset.NAME,
        #     LargeResource.ARCHIVE: "zip",
        #     LargeResource.URI: CatacxDataset.URL
        # })
        #
        # filepath = os.path.join(
        #     LargeResource.BASE_RESOURCE_DIR,
        #     CatacxDataset.DATASET_DIR,
        #     CatacxDataset.DATASET_FILE_NAME)
        #
        # return CatacxDataset.load_from_file(filepath, fields=fields)

    @staticmethod
    def load_from_file(path, fields=None):
        fields = fields if fields else CatacxDataset.get_default_fields()
        parser = CatacxDataset._get_catacx_parser()

        with open(path, mode='r') as f:
            ds_str = f.read()

        return HierarchicalDataset.from_json(ds_str, fields, parser)

    @staticmethod
    def _get_catacx_parser():
        example_factory = ExampleFactory(CatacxDataset.get_default_fields())

        def catacx_parser(raw, fields, depth):
            example = example_factory.from_dict(raw)

            # the catacx dataset has different names for fields containing the children
            # of a node depending on the depth of the parent node.
            if depth == 0:
                # children of root nodes are stored in the 'comments' field
                children = raw.get('comments')

            elif depth == 1:
                # children of comments are stored in the 'replies' field
                children = raw.get('replies')

            else:
                # posts don't have any children, so return an unit tuple
                children = ()

            return example, children

        return catacx_parser

    @staticmethod
    def get_sentence_tokenizer(key):
        def extractor_tokenizer(raw):
            # raw : list of lists of dicts containing the preprocessed data
            tokens = []
            for sentence in raw:
                # sentence : list of dicts containing the preprocessed data
                for token in sentence:
                    # tokend : dict containing preprocessed data
                    tokens.append(token.get(key))

            return tokens

        return extractor_tokenizer

    @staticmethod
    def get_default_fields():
        """
        Method returns a dict of default Catacx fields.

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

        # id_field = Field("id",
        #                  store_as_raw=True,
        #                  tokenize=False)

        sentiment_field = Field("sentiment",
                                store_as_raw=True,
                                tokenize=False,
                                custom_numericalize=float,
                                allow_missing_data=True)

        likes_cnt_field = Field("likes_cnt",
                                store_as_raw=True,
                                tokenize=False,
                                custom_numericalize=int)

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
                           allow_missing_data=True)

        emotions_field = MultilabelField("emotions",
                                         vocab=Vocab(specials=()),
                                         allow_missing_data=True)

        irony_field = Field("irony",
                            store_as_raw=True,
                            tokenize=False,
                            custom_numericalize=int,
                            allow_missing_data=True)

        speech_acts_field = MultilabelField("speech_acts",
                                            vocab=Vocab(specials=()),
                                            allow_missing_data=True)

        topics_field = MultilabelField("topics",
                                       vocab=Vocab(specials=()),
                                       allow_missing_data=True)

        cs_field = MultilabelField("cs",
                                   vocab=Vocab(specials=()),
                                   allow_missing_data=True)

        pos_tag_field = Field("pos_tags",
                              store_as_raw=False,
                              tokenizer=CatacxDataset.get_sentence_tokenizer("pos_tag"))

        lemma_field = Field("lemmas",
                            store_as_raw=False,
                            tokenizer=CatacxDataset.get_sentence_tokenizer("lemma"))

        parent_ids_field = Field("parent_ids",
                                 store_as_raw=False,
                                 tokenizer=CatacxDataset
                                 .get_sentence_tokenizer("parent_id"))

        tokens_field = Field("tokens",
                             store_as_raw=False,
                             tokenizer=CatacxDataset.get_sentence_tokenizer("token"))

        dependency_tags_field = Field("dependency_tags",
                                      store_as_raw=False,
                                      tokenizer=CatacxDataset.get_sentence_tokenizer(
                                          "dependency_tag"))

        token_id_field = Field("id_tags",
                               store_as_raw=False,
                               tokenizer=CatacxDataset.get_sentence_tokenizer("id"))

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
