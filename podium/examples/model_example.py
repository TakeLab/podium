"""Example how to use model on simple PauzaHR dataset."""

from podium.storage import Field, LargeResource, Vocab
from podium.datasets.iterator import Iterator
from podium.datasets.impl.pauza_dataset import PauzaHRDataset
from podium.models.impl.fc_model import ScikitMLPClassifier
from podium.models.impl.simple_trainers import SimpleTrainer
from podium.models import FeatureTransformer


def numericalize_pauza_rating(rating):
    """Function numericalizes pauza_hr dataset rating field"""
    label = round(float(rating) * 2) - 1
    return label


def label_extraction_fun(y_batch):
    """Label transform function that returns a 1-d array of rating labels."""
    return y_batch.Rating.ravel()


def feature_extraction_fn(x_batch):
    """Feature transform function that returns an matrix containing word indexes.
    Serves only as a simple demonstration."""
    x_tensor = x_batch.Text
    return x_tensor


def basic_pauza_hr_fields():
    """Function returns pauza-hr fields used for classification."""
    rating = Field(name="Rating", vocab=Vocab(specials=()), store_as_raw=True,
                   is_target=True, tokenize=False,
                   custom_numericalize=numericalize_pauza_rating)
    text = Field(name="Text", vocab=Vocab(), tokenizer='split',
                 language="hr", tokenize=True, store_as_raw=False,
                 fixed_length=100)
    return {"Text": text, "Rating": rating}


def pauza_mlp_example():
    """Adjustable example that demonstrates how to use pauzahr dataset
    with scikit MLP classifier using podium"""

    # Set the base repository directory
    # This directory will be used by podium to cache all LargeResources
    # like datasets and vectorizers loaded trough the LargeResource API
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"

    fields = basic_pauza_hr_fields()
    train_set, test_set = PauzaHRDataset.get_train_test_dataset(fields=fields)

    train_iter = Iterator(batch_size=100)

    model = ScikitMLPClassifier(
        classes=[i for i in range(len(fields["Rating"].vocab.itos))],
        verbose=True, hidden_layer_sizes=(50, 20), solver="adam")

    # Define a FeatureTranformer used to extract and transform feature matrices
    # from feature batches
    feature_transformer = FeatureTransformer(feature_extraction_fn)

    trainer = SimpleTrainer()
    trainer.train(model,
                  train_set,
                  iterator=train_iter,
                  feature_transformer=feature_transformer,
                  label_transform_fun=label_extraction_fun,
                  **{trainer.MAX_EPOCH_KEY: 10})

    test_batch_x, test_batch_y = next(train_iter(train_set).__iter__())
    x_test = feature_transformer.transform(test_batch_x)
    y_test = label_extraction_fun(test_batch_y)
    prediction = model.predict(X=x_test)
    print("Expected:\t", y_test, "\n",
          "Given:\t\t", prediction[model.PREDICTION_KEY])


if __name__ == "__main__":
    pauza_mlp_example()
