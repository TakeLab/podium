"""Exampel how to use simple PauzaHR dataset."""
from takepod.storage.large_resource import LargeResource
from takepod.datasets.pauza_dataset import PauzaHRDataset


if __name__ == "__main__":
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"
    train_set, test_set = PauzaHRDataset.get_train_test_dataset()
    print("train set {}".format(len(train_set)))
    print("test set {}".format(len(test_set)))
