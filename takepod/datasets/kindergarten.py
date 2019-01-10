import os
import json
from takepod.storage.example import Example
from takepod.storage.field import Field
from takepod.storage.vocab import Vocab
from takepod.datasets.catacx_comments_dataset import CatacxCommentsDataset

DIR = "~/Desktop/TakeLab/catacx_dataset/catacx_dataset.json"


def get_comments(dataset):
    for post in dataset:
        for comment in post["comments"]:
            yield comment

if __name__ == '__main__':
    file_dir = os.path.expanduser(DIR)
    ds = CatacxCommentsDataset(file_dir)
    ds.finalize_fields()
    pass
    for e in ds:
        print(e)