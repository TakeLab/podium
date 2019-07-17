import pytest
import yake

from takepod.storage.yake import YAKE

TEXT = """Sources tell us that Google is acquiring Kaggle, a platform that hosts
        data science and machine learning competitions. Details about the
        transaction remain somewhat vague, but given that Google is hosting
        its Cloud Next conference in San Francisco this week, the official
        announcement could come as early as tomorrow.  Reached by phone,
        Kaggle co-founder CEO Anthony Goldbloom declined to deny that the
        acquisition is happening. Google itself declined
        'to comment on rumors'.
        """


@pytest.fixture()
def keyword_data():
    return {
        "text": TEXT
    }


@pytest.mark.usefixtures("keyword_data")
def test_yake_en_wrapper_output(keyword_data):
    yake_original = yake.KeywordExtractor()
    yake_takepod = YAKE()

    output_original = [out[0] for out in
                       yake_original.extract_keywords(keyword_data["text"])]
    output_takepod = yake_takepod(keyword_data["text"])

    assert output_takepod == output_original
