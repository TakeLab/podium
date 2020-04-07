import pytest

pytest.importorskip("podium.preproc.yake")
from podium.preproc.yake import YAKE
import yake


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


KEYWORD_DATA = {
    "text": TEXT,
    "keywords": KEYWORDS
}




def test_yake_en_wrapper_output():
    yake = pytest.importorskip('yake')
    yake_original = yake.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.9,
        dedupFunc='seqm',
        windowsSize=1,
        top=20)

    yake_podium = YAKE(
        lan="en",
        n=3,
        dedupLim=0.9,
        dedupFunc='seqm',
        windowsSize=1,
        top=20
    )

    output_original = [kw for kw, _ in
                       yake_original.extract_keywords(KEYWORD_DATA["text"])]
    output_podium = yake_podium(KEYWORD_DATA["text"])

    assert output_podium == output_original


def test_yake_en_nondefault_wrapper_output():
    yake_original = yake.KeywordExtractor(n=2)
    yake_podium = YAKE(n=2)

    output_original = [kw for kw, _ in
                       yake_original.extract_keywords(KEYWORD_DATA["text"])]
    output_podium = yake_podium(KEYWORD_DATA["text"])

    assert output_podium == output_original
