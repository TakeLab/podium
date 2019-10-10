"""Example how to use model on simple PauzaHR dataset using the Experiment class."""

import numpy as np

from takepod.storage import Field, LargeResource, Vocab, \
    ExampleFormat
from takepod.datasets import Iterator
from takepod.datasets.impl.pauza_dataset import PauzaHRDataset
from takepod.models.impl.fc_model import ScikitMLPClassifier
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.models import Experiment, FeatureTransformer, SklearnTensorTransformerWrapper
from takepod.validation import k_fold_classification_metrics
from takepod.model_selection import grid_search
from takepod.storage.vectorizers.impl import NlplVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from takepod.pipeline import Pipeline


def numericalize_pauza_rating(rating):
    """Function numericalizes pauza_hr dataset rating field"""
    label = round(float(rating) * 2) - 1
    return label


def basic_pauza_hr_fields():
    """Function returns pauza-hr fields used for classification."""
    rating = Field(name="Rating", vocab=Vocab(specials=()),
                   is_target=True, tokenize=False,
                   custom_numericalize=numericalize_pauza_rating)

    text = Field(name="Text", vocab=Vocab(), tokenizer='split',
                 language="hr", tokenize=True, store_as_raw=False,
                 fixed_length=100)

    return {"Text": text, "Rating": rating}


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

    vectorizer = NlplVectorizer()
    vectorizer.load_vocab(vocab=fields["Text"].vocab)
    embedding_matrix = vectorizer.get_embedding_matrix(
        fields["Text"].vocab)

    def feature_transform_fn(x_batch):
        """Batch transform function that returns a mean of embedding vectors for every
            token in an Example"""
        x_tensor = np.take(embedding_matrix, x_batch.Text.astype(int), axis=0)
        x = np.mean(x_tensor, axis=1)
        return x

    tensor_transformer = SklearnTensorTransformerWrapper(StandardScaler())
    feature_transformer = FeatureTransformer(feature_transform_fn, tensor_transformer)

    experiment = Experiment(ScikitMLPClassifier,
                            trainer=trainer,
                            training_iterator_callable=train_iterator_provider,
                            feature_transformer=feature_transformer,
                            label_transform_fun=label_transform_fun)

    _, model_params, train_params = \
        grid_search(experiment,
                    test_dataset,
                    accuracy_score,
                    model_param_grid={'classes': ([i for i in range(num_of_classes)],),
                                      'hidden_layer_sizes': [(10,), (10, 10), (100,)]},
                    trainer_param_grid={SimpleTrainer.MAX_EPOCH_KEY: [2, 3, 4, 5]}
                    )

    experiment.set_default_model_args(**model_params)
    experiment.set_default_trainer_args(**train_params)

    accuracy, precision, recall, f1 = k_fold_classification_metrics(experiment,
                                                                    test_dataset,
                                                                    5,
                                                                    average='macro')

    print("Accuracy = {}\n"
          "Precision = {}\n"
          "Recall = {}\n"
          "F1 score = {}".format(accuracy, precision, recall, f1))

    experiment.fit(train_dataset)

    dataset_fields = {
        "Text": train_dataset.field_dict["Text"]
    }

    pipeline = Pipeline(dataset_fields,
                        ExampleFormat.XML,
                        feature_transformer,
                        experiment.model)

    example_good = "<Example><Text>Izvrstan, ogroman Zagrebački, " \
                   "dostava na vrijeme, ljubazno osoblje ...</Text></Example>"
    prediction = pipeline.predict_raw(example_good)
    print("Good example score: {}".format(prediction))

    example_bad = "<Example><Text>Hrana kasnila, dostavljac neljubazan, " \
                  "uzas...</Text></Example>"
    prediction = pipeline.predict_raw(example_bad)
    print("Bad example score: {}".format(prediction))


if __name__ == '__main__':
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"
    experiment_example()
