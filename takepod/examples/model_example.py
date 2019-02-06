"""Example how to use model on simple PauzaHR dataset."""
from takepod.storage.large_resource import LargeResource
from takepod.storage.field import Field
from takepod.storage.vocab import Vocab
from takepod.storage.iterator import Iterator
from takepod.datasets.pauza_dataset import PauzaHRDataset
from takepod.models.fc_model import ScikitMLPClassifier
from takepod.models.simple_trainers import SimpleTrainer


def numericalize_pauza_rating(rating):
    """Function numericalizes pauza_hr dataset rating field"""
    label = int(float(rating)*2)
    return label


def main():
    """Example that demonstrates how to use pauzahr dataset with scikit MLP
    classifier using podium"""
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"
    rating = Field(name="Rating", vocab=None, store_raw=True, is_target=True,
                   sequential=False,
                   custom_numericalize=numericalize_pauza_rating)
    text = Field(name="Text", vocab=Vocab(), tokenizer='split',
                 language="hr", sequential=True, store_raw=False,
                 fixed_length=100)
    fields = {"Text": text, "Rating": rating}

    train_set, test_set = PauzaHRDataset.get_train_test_dataset(fields=fields)
    print("train set {}".format(len(train_set)))
    print("test set {}".format(len(test_set)))

    train_iter = Iterator(dataset=train_set, batch_size=5)
    test_iter = Iterator(dataset=test_set, batch_size=5)

    model = ScikitMLPClassifier(classes=[i for i in range(1, 13)])
    trainer = SimpleTrainer(model=model)

    trainer.train(iterator=train_iter, **{trainer.MAX_EPOCH_KEY: 100})

    x_test, y_test = next(test_iter.__iter__())
    prediction = model.predict(X=x_test[0])
    print("Expected:", y_test[0], "given:", prediction[model.PREDICTION_KEY])

if __name__ == "__main__":
    main()
