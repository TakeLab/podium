"""
Example how to use model on simple PauzaHR dataset using the Experiment class.
"""
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from podium import Field, LabelField, Vocab
from podium.datasets import ExampleFormat, Iterator, PauzaHRDataset
from podium.storage import LargeResource
from podium.vectorizers.impl import NlplVectorizer
from podium.experimental.models import Experiment, FeatureTransformer, SklearnTensorTransformerWrapper
from podium.experimental.models.impl.fc_model import ScikitMLPClassifier
from podium.experimental.models.impl.simple_trainers import SimpleTrainer
from podium.experimental.model_selection import grid_search
from podium.experimental.pipeline import Pipeline
from podium.experimental.validation import k_fold_classification_metrics


def numericalize_pauza_rating(rating):
    """
    Function numericalizes pauza_hr dataset rating field.
    """
    label = round(float(rating) * 2) - 1
    return label


def basic_pauza_hr_fields():
    """
    Function returns pauza-hr fields used for classification.
    """
    rating = LabelField(
        name="Rating",
        pretokenize_hooks=[numericalize_pauza_rating]
    )

    text = Field(
        name="Text",
        numericalizer=Vocab(),
        tokenizer="split",
        keep_raw=False,
        fixed_length=100,
    )

    return {"Text": text, "Rating": rating}


def label_transform_fun(y_batch):
    return y_batch.Rating.ravel()


def experiment_example():
    """
    Example of setting up and using the Experiment class.
    """

    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"

    fields = basic_pauza_hr_fields()
    train_dataset, test_dataset = PauzaHRDataset.get_train_test_dataset(fields)

    num_of_classes = len(train_dataset.field_dict["Rating"].vocab.itos)

    vector_cache_path = os.path.join(
        LargeResource.BASE_RESOURCE_DIR, "experimet_example_nlpl_cache.txt"
    )

    vectorizer = NlplVectorizer(cache_path=vector_cache_path)
    vectorizer.load_vocab(vocab=fields["Text"].vocab)
    embedding_matrix = vectorizer.get_embedding_matrix(fields["Text"].vocab)

    def feature_transform_fn(x_batch):
        """
        Batch transform function that returns a mean of embedding vectors for
        every token in an Example.
        """
        x_tensor = np.take(embedding_matrix, x_batch.Text.astype(int), axis=0)
        x = np.mean(x_tensor, axis=1)
        return x

    trainer = SimpleTrainer()

    tensor_transformer = SklearnTensorTransformerWrapper(StandardScaler())
    feature_transformer = FeatureTransformer(feature_transform_fn, tensor_transformer)

    experiment = Experiment(
        ScikitMLPClassifier,
        trainer=trainer,
        feature_transformer=feature_transformer,
        label_transform_fn=label_transform_fun,
    )

    _, model_params, train_params = grid_search(
        experiment,
        test_dataset,
        accuracy_score,
        model_param_grid={
            "classes": ([i for i in range(num_of_classes)],),
            "hidden_layer_sizes": [(10,), (10, 10), (100,)],
        },
        trainer_param_grid={
            "max_epoch": [2, 3, 4],
            "iterator": [Iterator(batch_size=32), Iterator(batch_size=64)],
        },
    )

    experiment.set_default_model_args(**model_params)
    experiment.set_default_trainer_args(**train_params)

    accuracy, precision, recall, f1 = k_fold_classification_metrics(
        experiment, test_dataset, 5, average="macro"
    )

    print(
        f"Accuracy = {accuracy}\n"
        f"Precision = {precision}\n"
        f"Recall = {recall}\n"
        f"F1 score = {f1}"
    )

    experiment.fit(train_dataset)

    dataset_fields = {"Text": train_dataset.field_dict["Text"]}

    pipeline = Pipeline(
        fields=dataset_fields,
        example_format=ExampleFormat.XML,
        model=experiment.model,
        feature_transformer=feature_transformer,
    )

    example_good = (
        "<Example><Text>Izvrstan, ogroman Zagrebački, "
        "dostava na vrijeme, ljubazno osoblje ...</Text></Example>"
    )
    prediction = pipeline.predict_raw(example_good)
    print(f"Good example score: {prediction}")

    example_bad = (
        "<Example><Text>Hrana kasnila, dostavljac neljubazan, " "uzas...</Text></Example>"
    )
    prediction = pipeline.predict_raw(example_bad)
    print(f"Bad example score: {prediction}")


if __name__ == "__main__":
    experiment_example()
