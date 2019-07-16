"""Example how to use model on simple PauzaHR dataset."""
from functools import partial
import numpy as np

from takepod.storage import Field, LargeResource, Vocab, Iterator
from takepod.storage.vectorizer import BasicVectorStorage
from takepod.datasets.pauza_dataset import PauzaHRDataset
from takepod.models.fc_model import ScikitMLPClassifier
from takepod.models.simple_trainers import SimpleTrainer


def numericalize_pauza_rating(rating):
    """Function numericalizes pauza_hr dataset rating field"""
    label = round(float(rating) * 2)
    return label


def basic_batch_transform_fun(x_batch, y_batch):
    """Method transforms iterator batch to a
       numpy matrix that model accepts."""
    X = x_batch.Text
    y = y_batch.Rating.ravel()
    return X, y


def batch_transform_fun_vectorize_avg(x_batch, y_batch, embedding_matrix):
    """Method transforms iterator batch to a
       numpy matrix that model accepts."""
    x_numericalized = x_batch.Text.astype(int)
    embeddings = np.take(embedding_matrix, x_numericalized, axis=0)
    X = np.mean(embeddings, axis=1)
    y = y_batch.Rating.ravel()
    return X, y


def basic_pauza_hr_fields():
    """Function returns pauza-hr fields used for classification."""
    rating = Field(name="Rating", vocab=Vocab(specials=()), store_as_raw=True,
                   is_target=True, tokenize=False,
                   custom_numericalize=numericalize_pauza_rating)
    text = Field(name="Text", vocab=Vocab(), tokenizer='split',
                 language="hr", tokenize=True, store_as_raw=False,
                 fixed_length=100)
    return {"Text": text, "Rating": rating}


def pauza_mlp_example(
        fields, dataset,
        batch_transform_function=basic_batch_transform_fun):
    """Adjustable example that demonstrates how to use pauzahr dataset
    with scikit MLP classifier using podium

    Parameters
    ----------
    fields : dict
        fields dictionary
    dataset : tuple(Dataset, Dataset)
        train, test dataset tuple
    batch_transform_function: callable
        function that know how to transform input batch to model input
    """
    train_set, test_set = dataset

    train_iter = Iterator(dataset=train_set, batch_size=100)
    test_iter = Iterator(dataset=test_set, batch_size=10)

    model = ScikitMLPClassifier(
        classes=[i for i in range(0, len(fields["Rating"].vocab.itos))],
        verbose=True, hidden_layer_sizes=(50, 20), solver="adam")
    trainer = SimpleTrainer()

    trainer.train(model, iterator=train_iter, **{
        trainer.MAX_EPOCH_KEY: 10,
        trainer.BATCH_TRANSFORM_FUN_KEY: batch_transform_function})

    x_test, y_test = batch_transform_function(*next(test_iter.__iter__()))
    prediction = model.predict(X=x_test)
    print("Expected:", y_test,
          "given:", prediction[model.PREDICTION_KEY])


def basic_pauza_example():
    """Basic pauza_hr example that tries to classify comments to ratings
    based on word numericalization."""
    fields = basic_pauza_hr_fields()
    dataset = PauzaHRDataset.get_train_test_dataset(fields=fields)
    pauza_mlp_example(
        fields=fields, dataset=dataset,
        batch_transform_function=basic_batch_transform_fun)


def vectorized_pauza_example():
    """Pauza_hr example that tries to classify comments to ratings based on
    averaged word vectors inside comments."""
    fields = basic_pauza_hr_fields()
    dataset = PauzaHRDataset.get_train_test_dataset(fields=fields)

    vectorizer = BasicVectorStorage(
        path="downloaded_datasets/tweeterVectors.txt")
    vectorizer.load_vocab(vocab=fields["Text"].vocab)
    embedding_matrix = vectorizer.get_embedding_matrix(
        fields["Text"].vocab)

    batch_transform = partial(
        batch_transform_fun_vectorize_avg,
        embedding_matrix=embedding_matrix)

    pauza_mlp_example(
        fields=fields, dataset=dataset,
        batch_transform_function=batch_transform)


if __name__ == "__main__":
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"
    vectorized_pauza_example()
