from takepod.models.base_model import SupervisedModel
from takepod.preproc.transform import (
    make_bow_vector,
    categories_to_int
)
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

        y = categories_to_int(y)
        iternum = min(len(X), 1000)
        for ep in range(5):
            total_loss = 0
            for i in range(iternum):

                self.model.zero_grad()
                j = random.randint(0, len(X) - 1)
                vector = make_bow_vector(X[j], word_to_ix)
                vector = torch.Tensor(vector).view(1, -1)

                log_probs = self.model(vector)
                target = torch.LongTensor([y[j]])
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
                bow_vec = torch.Tensor(bow_vec).view(1, -1)
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
