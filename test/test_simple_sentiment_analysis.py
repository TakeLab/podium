from takepod.models.simple_sentiment_analysis import SimpleSentimentAnalysisModel
from takepod.preproc.transform import create_word_to_index
import torch


def test_full_sentiment_analysis_model():
    X_train, y_train = ['Positive data', 'Bad data'], ['Positive', 'Negative']
    X_test, y_test = ['Positive stuff', 'Bad stuff'], ['Positive', 'Negative']

    word_to_ix = create_word_to_index(X_train + X_test)
    assert word_to_ix == {
        'Positive': 0,
        'data': 1,
        'Bad': 2,
        'stuff': 3
    }
    sa = SimpleSentimentAnalysisModel(vocab_size=len(word_to_ix))
    sa.train(X_train, y_train, word_to_ix=word_to_ix)
    predicted = sa.test(X_test, word_to_ix=word_to_ix)
    # not checking values since ML is involved
    # checking only the shape
    assert predicted.shape == torch.Size([2, 1])
