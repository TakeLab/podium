"""Module contains simple sentiment analysis model."""
import random
import torch.nn.functional as F
from torch import optim
from torch import nn
import torch
from takepod.models.contrib.base_model import SupervisedModel
from takepod.preproc.transform import (
    make_bow_vector,
    categories_to_int
)


class SimpleSentimentAnalysisModel(SupervisedModel):
    """Simple supervised trainable model
    for sentiment analysis

    Datasets: [IMDB]
    Languages: [English]
    """

    def __init__(self, **kwargs):
        self._hyperparameters = {
            "vocab_size": kwargs['vocab_size'],
            "num_labels": 2,
        }
        self.model = RNN(
            self.hyperparameters["vocab_size"],
            self.hyperparameters["num_labels"]
        )

    def train(self, data, labels, **kwargs):
        """Trains the sentiment analysis model

        Parameters
        ----------
        data : list
            List of unpreprocesed input data
        labels : list
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

        labels = categories_to_int(labels)
        iternum = min(len(data), 1000)
        for epoch in range(5):
            total_loss = 0
            for _ in range(iternum):

                self.model.zero_grad()
                j = random.randint(0, len(data) - 1)
                vector = make_bow_vector(data[j], word_to_ix)
                vector = torch.Tensor(vector).view(1, -1)

                log_probs = self.model(vector)
                target = torch.LongTensor([labels[j]])
                loss = loss_function(log_probs, target)
                total_loss += loss

                loss.backward()
                optimizer.step()
            print("loss in epoch {}: {}".format(epoch, total_loss))
        return total_loss

    def test(self, data, **kwargs):
        """Predict sentiment for given data

        Parameters
        ----------
        data : list
            List of unpreprocesed input data
        **kwargs : dict
            Additional key-value parameters to save on resources

        Returns
        -------
        predicted : torch.Tensor
            Predicted output labels
        """
        word_to_ix = kwargs["word_to_ix"]
        predicted = torch.zeros(len(data), 1)
        with torch.no_grad():
            for i in range(len(data)):
                bow_vec = make_bow_vector(data[i], word_to_ix)
                bow_vec = torch.Tensor(bow_vec).view(1, -1)
                log_probs = self.model(bow_vec)
                predicted[i] = torch.argmax(log_probs)
        return predicted

    # TODO for Bayesian Optimization there should
    # a clean way to access hyperparameters (class type would help)
    @property
    def hyperparameters(self):
        """Method for obtaining model hyperparameters.

        Returns
        -------
        hyperparameters : array-like
            array of hyperparameters
        """
        return self._hyperparameters


class RNN(nn.Module):
    """Simple RNN model class."""
    def __init__(self, vocab_size, num_labels):
        """RNN model constructor.

        Parameters
        ----------
        vocab_size : int
            vocabular size
        num_labels : int
            number of labels
        """
        super().__init__()
        self.fc_layer = nn.Linear(vocab_size, num_labels)

    def forward(self, x):
        """Method calculates model forward pass.

        Parameters
        ----------
        x : array-like
            input data

        Returns
        -------
        output : array-like
            model output for given data

        """
        return F.log_softmax(self.fc_layer(x), dim=1)
