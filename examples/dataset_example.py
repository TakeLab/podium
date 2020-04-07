"""Example how to use simple PauzaHR dataset."""
import dill

from podium.storage.resources.large_resource import LargeResource
from podium.datasets.impl.pauza_dataset import PauzaHRDataset

if __name__ == "__main__":
    # for large resource settings see
    # https://github.com/mtutek/podium/wiki/Large-resources
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"

    train_set, test_set = PauzaHRDataset.get_train_test_dataset()
    print("train set {}".format(len(train_set)))
    print("test set {}".format(len(test_set)))

    # save dataset
    with open("dataset.pkl", "wb") as f:
        dill.dump(train_set, f)

    # load dataset
    with open("dataset.pkl", "rb") as f:
        loaded_train_set = dill.load(f)
