"""
Module contains IMDB Large Movie Review Dataset
Dataset webpage: http://ai.stanford.edu/~amaas/data/sentiment/

When using this dataset, please cite:
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and
               Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for
               Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

import os
from takepod.storage import (dataset, Example, Field, Vocab, LargeResource)


class BasicSupervisedImdbDataset():
    NAME = "imdb"
    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    DATASET_DIR = os.path.join("imdb", "aclImdb")
    ARCHIVE_TYPE = "tar"
    TRAIN_DIR = "train"
    TEST_DIR = "test"

    def __init__(self):
        LargeResource(**{
            LargeResource.RESOURCE_NAME: BasicSupervisedImdbDataset.NAME,
            LargeResource.ARCHIVE: BasicSupervisedImdbDataset.ARCHIVE_TYPE,
            LargeResource.URI: BasicSupervisedImdbDataset.URL})
