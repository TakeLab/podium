from podium.arrow import ArrowDataset
from podium.storage import Field, Vocab


class PandoraDataset(ArrowDataset):
    @staticmethod
    def get_default_fields():

        fields = [
            Field("author", tokenizer=None, store_as_raw=True, is_numericalizable=False),
            Field(
                "author_flair_text",
                tokenizer=None,
                store_as_raw=True,
                is_numericalizable=False,
            ),
            Field("body", tokenizer=str.split, vocab=Vocab()),
            Field("downs", tokenizer=None, store_as_raw=True, custom_numericalize=float),
            Field(
                "created_utc", tokenizer=None, store_as_raw=True, is_numericalizable=False
            )
            # TODO rest of fields
        ]

        return {field.name: field for field in fields}

    @staticmethod
    def load_dataset(comments_file, fields=None, cache_path=None, data_types=None):
        fields = fields if fields else PandoraDataset.get_default_fields()
        return ArrowDataset.from_tabular_file(
            comments_file, "csv", fields, cache_path=cache_path, data_types=data_types
        )
