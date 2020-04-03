import pytest

from podium.preproc.yake import YAKE

TEXT = """Sources tell us that Google is acquiring Kaggle, a platform that hosts
        data science and machine learning competitions. Details about the
        transaction remain somewhat vague, but given that Google is hosting
        its Cloud Next conference in San Francisco this week, the official
        announcement could come as early as tomorrow.  Reached by phone,
        Kaggle co-founder CEO Anthony Goldbloom declined to deny that the
        acquisition is happening. Google itself declined
        'to comment on rumors'.
        """

KEYWORDS = ['machine learning competitions',
            'hosts data science',
            'learning competitions',
            'platform that hosts',
            'hosts data',
            'data science',
            'science and machine',
            'machine learning',
            'san francisco',
            'ceo anthony goldbloom',
            'google',
            'google is acquiring',
            'acquiring kaggle',
            'francisco this week',
            'hosting its cloud',
            'cloud next conference',
            'conference in san',
            'anthony goldbloom declined',
            'co-founder ceo anthony',
            'kaggle co-founder ceo']


@pytest.fixture()
def keyword_data():
    return {
        "text": TEXT,
        "keywords": KEYWORDS
    }


@pytest.mark.usefixtures("keyword_data")
def test_yake_en_wrapper_output(keyword_data):
    yake = pytest.importorskip('yake')
    yake_original = yake.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.9,
        dedupFunc='seqm',
        windowsSize=1,
        top=20)

    yake_takepod = YAKE(
        lan="en",
        n=3,
        dedupLim=0.9,
        dedupFunc='seqm',
        windowsSize=1,
        top=20
    )

    output_original = [kw for kw, _ in
                       yake_original.extract_keywords(keyword_data["text"])]
    output_takepod = yake_takepod(keyword_data["text"])

    assert output_takepod == output_original


# @pytest.mark.usefixtures("keyword_data")
# def test_yake_en_nondefault_wrapper_output(keyword_data):
#     yake = pytest.importorskip('yake')
#     yake_original = yake.KeywordExtractor(n=2)
#     yake_takepod = YAKE(n=2)
#
#     output_original = [kw for kw, _ in
#                        yake_original.extract_keywords(keyword_data["text"])]
#     output_takepod = yake_takepod(keyword_data["text"])
#
#     assert output_takepod == output_original

# @pytest.mark.usefixtures("keyword_data")
# def test_yake_en_default_output(keyword_data):
#     pytest.importorskip('yake')
#     yake = YAKE()
#     output_kws = yake(keyword_data["text"])
#
#     assert output_kws == keyword_data["keywords"]
