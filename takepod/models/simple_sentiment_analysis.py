from takepod.models.base_model import SupervisedModel
from takepod.dataload.load_imdb import load_data
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import random


class SimpleSentimentAnalysisModel(SupervisedModel):
    """Simple supervised trainable model
    for sentiment analysis

    Datasets: [IMDB]
    Languages: [English]
    """

    def __init__(self, *args, **kwargs):
        self._hyperparameters = {
            "vocab_size": kwargs['vocab_size'],
            "num_labels": 2,
        }
        self.model = RNN(
            self.hyperparameters["vocab_size"],
            self.hyperparameters["num_labels"]
        )

    def train(self, X, y, **kwargs):
        """Trains the sentiment analysis model

        Parameters
        ----------
        X : list
            List of unpreprocesed input data
        y : list
            List of unpreprocessed labels
        **kwargs : dict
            Additional key-value parameters to save on resources

        Returns
        -------
        total_loss : torch.Tensor
            Sum of negative log likelyhood
            loss for the final training epoch
        """
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        word_to_ix = kwargs["word_to_ix"]
        # randomly take stuff from the train set instead of
        # something sensible like batch or online sequential
        for ep in range(5):
            total_loss = 0
            for i in range(1000):

                self.model.zero_grad()
                j = random.randint(0, len(X))
                vector = make_bow_vector(X[j], word_to_ix)

                log_probs = self.model(vector)
                target = torch.LongTensor([round(y[j])])
                loss = loss_function(log_probs, target)
                total_loss += loss

                loss.backward()
                optimizer.step()
            print("loss in epoch {}: {}".format(ep, total_loss))
        return total_loss

    def test(self, X, **kwargs):
        """Predict sentiment for data X

        Parameters
        ----------
        X : list
            List of unpreprocesed input data
        **kwargs : dict
            Additional key-value parameters to save on resources

        Returns
        -------
        predicted : torch.Tensor
            Predicted output labels
        """
        word_to_ix = kwargs["word_to_ix"]
        predicted = torch.zeros(len(X), 1)
        with torch.no_grad():
            for i in range(len(X)):
                bow_vec = make_bow_vector(X[i], word_to_ix)
                log_probs = self.model(bow_vec)
                predicted[i] = torch.argmax(log_probs)
        return predicted

    # TODO for Bayesian Optimization there should
    # a clean way to access hyperparameters (class type would help)
    @property
    def hyperparameters(self):
        return self._hyperparameters


class RNN(nn.Module):

    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.fc = nn.Linear(vocab_size, num_labels)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)


# TODO move to preprocessing
def make_bow_vector(document, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in document.split(' '):

        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


# TODO this should be a part of preprocessing
# this should be a pre
def create_word_to_index(data):
    word_to_ix = {}
    for document in data:
        for word in document.split(' '):
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def test_run():
    # TODO there should be an execution model pattern of
    # 1. preprocess
    # 2. train
    # 3. test
    # 4. evaluate
    # ML models should adhere to this interface
    # preprocess
    # delete this once established
    X_train, y_train = load_data(train=True)
    X_test, y_test = load_data(train=False)
    word_to_ix = create_word_to_index(X_train + X_test)
    # init
    sa = SimpleSentimentAnalysisModel(vocab_size=len(word_to_ix))
    # train
    sa.train(X_train, y_train, word_to_ix=word_to_ix)
    # test
    predicted = sa.test(X_test[0:100], word_to_ix=word_to_ix)
    # evaluate below
    return predicted
