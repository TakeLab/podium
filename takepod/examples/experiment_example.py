"""Example how to use model on simple PauzaHR dataset using the Experiment class."""

from functools import partial
import numpy as np

from takepod.storage import (Field, LargeResource, Vocab,
                             BasicVectorStorage)
from takepod.datasets import Iterator, SingleBatchIterator
from takepod.datasets.impl.pauza_dataset import PauzaHRDataset
from takepod.models.impl.fc_model import ScikitMLPClassifier
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.models import Experiment
from takepod.validation import k_fold_multiclass_metrics


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


def feature_transform_mean_fun(x_batch, embedding_matrix):
    """Batch transform function that returns a mean of embedding vectors for every
    token in an Example"""
    x_tensor = np.take(embedding_matrix, x_batch.Text.astype(int), axis=0)
    x = np.mean(x_tensor, axis=1)
    return x


def basic_feature_transform_fun(x_batch):
    """Method transforms iterator batch to a
       numpy matrix that model accepts."""
    return x_batch.Text

def label_transform_fun(y_batch):
    return y_batch.Rating.ravel()

def experiment_example():
    """Example of setting up and using the Experiment class.
    """
    fields = basic_pauza_hr_fields()
    train_dataset, test_dataset = PauzaHRDataset.get_train_test_dataset(fields)

    num_of_classes = len(train_dataset.field_dict["Rating"].vocab.itos)

    trainer = SimpleTrainer()

    def train_iterator_provider(dataset):
        return Iterator(dataset, shuffle=True)

    vectorizer = BasicVectorStorage(path="downloaded_datasets/tweeterVectors.txt")
    vectorizer.load_vocab(vocab=fields["Text"].vocab)
    embedding_matrix = vectorizer.get_embedding_matrix(
        fields["Text"].vocab)

    feature_transform = partial(feature_transform_mean_fun, embedding_matrix=embedding_matrix)

    experiment = Experiment(ScikitMLPClassifier,
                            trainer,
                            train_iterator_provider,
                            None,
                            feature_transform,
                            label_transform_fun)

    print(k_fold_multiclass_metrics(experiment,
                                    test_dataset,
                                    5))

    #
    # experiment.fit(train_dataset,
    #                model_kwargs={
    #                    "classes": np.arange(0, num_of_classes),
    #                    "hidden_layer_sizes": (50, 20),
    #                    "solver": "adam"
    #                },
    #                trainer_kwargs={
    #                    SimpleTrainer.MAX_EPOCH_KEY: 5
    #                })
    #
    # test_dataset_slice = test_dataset
    #
    # _, true_values = next(iter(SingleBatchIterator(test_dataset_slice)))
    # true_values = true_values.Rating.ravel()
    #
    # predicted_values = experiment.predict(test_dataset_slice)
    # predicted_values = predicted_values.ravel()
    #
    # accuracy = np.count_nonzero(true_values == predicted_values) / len(predicted_values)
    #
    # print("Accuracy on the test set: ", accuracy)




if __name__ == '__main__':
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"
    experiment_example()
