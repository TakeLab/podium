"""Example how to use model on simple PauzaHR dataset."""
from functools import partial
import numpy as np

from takepod.storage import Field, LargeResource, Vocab, Iterator, SingleBatchIterator, BasicVectorStorage
from takepod.datasets.pauza_dataset import PauzaHRDataset
from takepod.models.fc_model import ScikitMLPClassifier
from takepod.models.simple_trainers import SimpleTrainer
from takepod.models import Experiment


def numericalize_pauza_rating(rating):
    """Function numericalizes pauza_hr dataset rating field"""
    label = round(float(rating) * 2) - 1
    return label


def basic_pauza_hr_fields():
    """Function returns pauza-hr fields used for classification."""
    rating = Field(name="Rating", vocab=Vocab(specials=()), store_as_raw=True,
                   is_target=True, tokenize=False,
                   custom_numericalize=numericalize_pauza_rating)

    text = Field(name="Text", vocab=Vocab(), tokenizer='split',
                 language="hr", tokenize=True, store_as_raw=False,
                 fixed_length=100)

    return {"Text": text, "Rating": rating}


def batch_transform_mean(x_batch, y_batch, embedding_matrix):
    x_tensor = np.take(embedding_matrix, x_batch.Text.astype(int), axis=0)
    x = np.mean(x_tensor, axis=1)
    y = np.ravel(y_batch.Rating)
    return x, y


def basic_batch_transform_fun(x_batch, y_batch):
    """Method transforms iterator batch to a
       numpy matrix that model accepts."""
    X = x_batch.Text
    y = y_batch.Rating.ravel()
    return X, y


def PauzaHR_experiment_example():
    fields = basic_pauza_hr_fields()
    train_dataset, test_dataset = PauzaHRDataset.get_train_test_dataset(fields)

    num_of_classes = len(train_dataset.field_dict["Rating"].vocab.itos)
    model = ScikitMLPClassifier(np.arange(num_of_classes))

    trainer = SimpleTrainer()

    def train_iterator_provider(dataset):
        return Iterator(dataset, shuffle=True)

    experiment = Experiment(model,
                            trainer,
                            basic_batch_transform_fun,
                            train_iterator_provider)

    experiment.fit(train_dataset,
                   model_kwargs={
                       "hidden_layer_sizes": (50, 20),
                       "solver": "adam"
                   },
                   trainer_kwargs={
                       SimpleTrainer.MAX_EPOCH_KEY: 5
                   })


if __name__ == '__main__':
    PauzaHR_experiment_example()
