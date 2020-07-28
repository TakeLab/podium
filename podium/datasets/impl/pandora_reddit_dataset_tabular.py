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
        # TODO add real fields
        id_field = Field('id', tokenize=False, store_as_raw=True,
                         is_numericalizable=False)

        return {
            'id': id_field
        }
