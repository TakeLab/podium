"""
Example how to use simple PauzaHR dataset.
"""
import dill

from podium.datasets import SST
from podium.storage import LargeResource


if __name__ == "__main__":
    # for large resource settings see
    # https://github.com/mttk/podium/wiki/Large-resources
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"

    train_set, _, test_set = SST.get_dataset_splits()
    print(f"train set {len(train_set)}")
    print(f"test set {len(test_set)}")

    # save dataset
    with open("dataset.pkl", "wb") as f:
        dill.dump(train_set, f)

    # load dataset
    with open("dataset.pkl", "rb") as f:
        loaded_train_set = dill.load(f)
