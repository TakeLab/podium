from podium.datasets import TabularDataset
from podium.storage import Field


class PandoraDatasetTabular(TabularDataset):

    @staticmethod
    def get_dataset(comments_path, fields=None):
        if fields is None:
            fields = PandoraDatasetTabular.get_default_fields()

        return PandoraDatasetTabular(comments_path, 'csv', fields)

    @staticmethod
    def get_default_fields():
        author_field = Field('author', tokenize=False, store_as_raw=True,
                             is_numericalizable=False)

        author_flair_text_field = Field('author_flair_text', tokenize=False, store_as_raw=True,
                                  is_numericalizable=False)

        downs_field = Field('downs', custom_numericalize=float)

        created_utc_field = Field('created_utc', custom_numericalize=float)

        subreddit_id_field = Field('subreddit_id', tokenize=False, store_as_raw=True,
                                   is_numericalizable=False)

        id_field = Field('id', tokenize=False, store_as_raw=True,
                         is_numericalizable=False)

        body_field = Field('body')

        #  TODO add other fields

        return {
            'author': author_field,
            'author_flair_text': author_flair_text_field,
            'body': body_field,
            'id': id_field,
            'downs': downs_field,
            'created_utc': created_utc_field,
            'subreddit_id': subreddit_id_field
        }
