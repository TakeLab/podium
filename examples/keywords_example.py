"""Example shows how to add keyword extraction features to a dataset."""
from podium.datasets import Dataset
from podium.preproc.yake import YAKE
from podium.storage import ExampleFactory, Field


class DummyDataset(Dataset):
    """Dummmy dataset.
    """
    TEXT_FIELD_NAME = "text"

    def __init__(self, texts, fields):
        """
        Dataset constructor.

        Parameters
        ----------
        texts : list of str
            list of document represented as strings
        """
        example_factory = ExampleFactory(fields)
        examples = [example_factory.from_dict({DummyDataset.TEXT_FIELD_NAME: text})
                    for text in texts]
        super(DummyDataset, self).__init__(
            **{"examples": examples, "fields": fields})


def keyword_extraction_main():
    """Function creates a dummmy keyword extraction dataset in Croatian language
    and extracts the keywords. The created dataset demonstrates how to map the input
    text to two fields: tokens (tokenized using str.split) and keywords (extracted
    using YAKE)."""

    sample_texts = ["""Karijera u turizmu Pjevačica Renata Končić Minea već dva tjedna radi kao
                       prodajni predstavnik u odjelu korporativnog poslovanja turističke
                       agencije Adriatica.net, no zbog toga se neće, tvrdi 27-godišnja
                       Zagrepčanka, odreći glazbe. Minea se prije deset godina, kad je
                       počinjala pjevačku karijeru i imala veliki hit 'Vrapci i komarci',
                       ispisala iz ekonomske škole, a poslije je maturirala na dopisnoj
                       birotehničkoj školi. (Marijana Marinović/Matko Stanković)""",
                    """MISS UNIVERSE TESTIRANA NA AIDS JOHANNESBURG - Aktualna Miss Universe
                       podvrgnula se u utorak testu na AIDS u jednoj bolnici u
                       Johannesburgu i izrazila nadu da će njezina popularnost uvjeriti
                       druge ljude da učine isto. Brineta plavih očiju Natalie Gtebova,
                       23-godišnja Kanađanka rođena u Rusiji, izjavila je da želi
                       iskoristiti svoju titulu za podizanje svjesnosti i borbe protiv
                       stigme koja okružuju tu bolest. »Mislim da će činjenica da sam se
                       javno testirala govoriti vrlo mnogo. To će ohrabriti puno mladih
                       žena da učine isto«, rekla je ona. Južnoafrička Republika bilježi
                       najveći broj zaraženih HlV-om - više od pet milijuna ljudi. (H)""",
                    ]

    tokens = Field("tokens",
                   tokenizer="split",
                   vocab=None,
                   tokenize=True,
                   store_as_raw=True,
                   store_as_tokenized=False
                   )
    kws = Field("keywords",
                tokenizer=YAKE('hr'),
                vocab=None,
                tokenize=True,
                store_as_raw=True,
                store_as_tokenized=False
                )
    fields = {DummyDataset.TEXT_FIELD_NAME: (tokens, kws)}
    dummy_dataset = DummyDataset(texts=sample_texts, fields=fields)
    keywords = [ex.keywords[1] for ex in dummy_dataset]
    print(*keywords, sep='\n')


if __name__ == "__main__":
    keyword_extraction_main()
