"""Example how to use tfidf with svm on simple Imdb dataset."""
from sklearn.metrics import accuracy_score

from takepod.storage.vectorizers.tfidf import TfIdfVectorizer
from takepod.models.impl import ScikitLinearSVCModel
from takepod.models import AbstractSupervisedModel, FeatureTransformer
from takepod.datasets import BasicSupervisedImdbDataset
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.storage import LargeResource
from takepod.datasets.iterator import Iterator, SingleBatchIterator


def tfidf_svm_example_main():
    """Function obtains imdb dataset and then trains scikit svc linear model by using
    tfidf as input."""
    train_set, test_set = BasicSupervisedImdbDataset.get_train_test_dataset()

    train_iter = Iterator(batch_size=len(train_set))
    test_iter = SingleBatchIterator()

    tfidf_vectorizer = TfIdfVectorizer()
    tfidf_vectorizer.fit(dataset=train_set, field=train_set.field_dict["text"])

    def feature_extraction_fn(x_batch):
        return tfidf_vectorizer.transform(x_batch.text)

    def label_extraction_fn(y_batch):
        return y_batch.label.ravel()

    feature_transformer = FeatureTransformer(feature_extraction_fn)

    model = ScikitLinearSVCModel()
    trainer = SimpleTrainer()

    trainer.train(model=model,
                  dataset=train_set,
                  iterator=train_iter,
                  feature_transformer=feature_transformer,
                  label_transform_fun=label_extraction_fn,
                  **{trainer.MAX_EPOCH_KEY: 1})

    x_batch, y_batch = next(test_iter(train_iter).__iter__())
    x_train = feature_transformer.transform(x_batch)
    y_train = label_extraction_fn(y_batch)
    prediction_train = model.predict(X=x_train)[AbstractSupervisedModel.PREDICTION_KEY]
    print(x_train.shape, y_train.shape, prediction_train.shape)
    print(accuracy_score(y_true=y_train, y_pred=prediction_train))

    x_batch, y_batch = next(test_iter(test_set).__iter__())
    x_test = feature_transformer.transform(x_batch)
    y_test = label_extraction_fn(y_batch)
    prediction_test = model.predict(X=x_test)[AbstractSupervisedModel.PREDICTION_KEY]
    print(x_test.shape, y_test.shape, prediction_test.shape)
    print(accuracy_score(y_true=y_test, y_pred=prediction_test))


if __name__ == "__main__":
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"
    tfidf_svm_example_main()
