"""Example how to use tfidf with svm on simple Imdb dataset."""
from functools import partial
from sklearn.metrics import accuracy_score

from takepod.storage.vectorizers.tfidf import TfIdfVectorizer
from takepod.models import AbstractSupervisedModel
from takepod.models.impl.svm_model import ScikitLinearSVCModel

from takepod.datasets import BasicSupervisedImdbDataset
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.storage import LargeResource
from takepod.datasets.iterator import Iterator


def batch_transform_tfidf(x_batch, y_batch, tfidf_vectorizer):
    """Method transforms iterator batch to a
       numpy matrix that model accepts."""
    X = tfidf_vectorizer.transform(x_batch.text)
    y = y_batch.label.ravel()
    return X, y


def tfidf_svm_example_main():
    """Function obtains imdb dataset and then trains scikit svc linear model by using
    tfidf as input."""
    train_set, test_set = BasicSupervisedImdbDataset.get_train_test_dataset()

    train_iter = Iterator(dataset=train_set, batch_size=len(train_set))
    test_iter = Iterator(dataset=test_set, batch_size=len(test_set))

    tfidf_vectorizer = TfIdfVectorizer()
    tfidf_vectorizer.fit(dataset=train_set, field=train_set.field_dict["text"])

    batch_transform = partial(
        batch_transform_tfidf,
        tfidf_vectorizer=tfidf_vectorizer)

    model = ScikitLinearSVCModel()
    trainer = SimpleTrainer(model=model)

    trainer.train(iterator=train_iter, batch_transform=batch_transform,
                  **{trainer.MAX_EPOCH_KEY: 1})

    x_train, y_train = batch_transform(*next(train_iter.__iter__()))
    prediction_train = model.predict(X=x_train)[AbstractSupervisedModel.PREDICTION_KEY]
    print(x_train.shape, y_train.shape, prediction_train.shape)
    print(accuracy_score(y_true=y_train, y_pred=prediction_train))

    x_test, y_test = batch_transform(*next(test_iter.__iter__()))
    prediction_test = model.predict(X=x_test)[AbstractSupervisedModel.PREDICTION_KEY]
    print(x_test.shape, y_test.shape, prediction_test.shape)
    print(accuracy_score(y_true=y_test, y_pred=prediction_test))


if __name__ == "__main__":
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"
    tfidf_svm_example_main()
