"""Example how to use simple PauzaHR dataset."""
import dill
import logging

from takepod.storage.large_resource import LargeResource
from takepod.datasets.pauza_dataset import PauzaHRDataset

if __name__ == "__main__":
    # for large resource settings see
    # https://github.com/FilipBolt/takepod/wiki/Large-resources
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"

    # for logging settings see
    # https://github.com/FilipBolt/takepod/wiki/Logging
    logging.config.fileConfig(
        fname='logging.ini', disable_existing_loggers=False)

    train_set, test_set = PauzaHRDataset.get_train_test_dataset()
    print("train set {}".format(len(train_set)))
    print("test set {}".format(len(test_set)))

    # save dataset
    with open("dataset.pkl", "wb") as f:
        dill.dump(train_set, f)

    # load dataset
    with open("dataset.pkl", "rb") as f:
        loaded_train_set = dill.load(f)
