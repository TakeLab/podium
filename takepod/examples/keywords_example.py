from takepod.storage import ExampleFactory, Field, dataset
from takepod.storage.yake import YAKE


class KWDummyDataset(dataset.Dataset):
    """Dummmy Keyword Extraction dataset. Consists of a single example (document)
       with a single field (raw text).

    Attributes
    ----------
    TEXT_FIELD_NAME : str
        name of the field containing the document text
    """

    TEXT_FIELD_NAME = "text"

    def __init__(self, text):
        """
        Dataset constructor.

        Parameters
        ----------
        text : str
            single document text contents
        """
        fields = KWDummyDataset.get_default_fields()
        unpacked_fields = dataset.unpack_fields(fields=fields)
        example_factory = ExampleFactory(fields)
        examples = [example_factory.from_dict({"text": text})]
        super(KWDummyDataset, self).__init__(
            **{"examples": examples, "fields": unpacked_fields})

    @staticmethod
    def get_default_fields():
        """Method returns default keyword extraction fields: text.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """
        text = Field(KWDummyDataset.TEXT_FIELD_NAME,
                     tokenizer='split',
                     language='en',
                     vocab=None,
                     tokenize=False,
                     store_as_raw=True,
                     store_as_tokenized=False
                     )
        return {KWDummyDataset.TEXT_FIELD_NAME: text}


def keyword_extraction_main(lang):
    texts = {"en": """Sources tell us that Google is acquiring Kaggle, a platform that hosts
                      data science and machine learning competitions. Details about the
                      transaction remain somewhat vague, but given that Google is hosting
                      its Cloud Next conference in San Francisco this week, the official
                      announcement could come as early as tomorrow.  Reached by phone,
                      Kaggle co-founder CEO Anthony Goldbloom declined to deny that the
                      acquisition is happening. Google itself declined
                      'to comment on rumors'.
                   """,
             "hr": """kljucna rijec"""}

    dataset = KWDummyDataset(texts[lang])
    kw_extractor = YAKE(lang)
    keywords = kw_extractor.transform(dataset[0])
    return keywords


if __name__ == "__main__":
    keywords_en = keyword_extraction_main("en")
    print(*keywords_en, sep='\n')
    # keywords_hr = keyword_extraction_main("hr")
    # print(*keywords_hr, sep='\n')
