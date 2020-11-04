"""
Home of the TakeLab Podium project. Podium is a Python machine learning library
that helps users to accelerate use of natural language processing models.

See http://takelab.fer.hr/podium/ for complete documentation.
"""
import logging

from .datasets import (
    BucketIterator,
    Dataset,
    HierarchicalDataset,
    HierarchicalDatasetIterator,
    Iterator,
    SingleBatchIterator,
    TabularDataset,
)
from .storage import Field, LabelField, MultilabelField, MultioutputField, Vocab


__all__ = [
    "Field",
    "LabelField",
    "MultilabelField",
    "MultioutputField",
    "Vocab",
    "Dataset",
    "TabularDataset",
    "HierarchicalDataset",
    "Iterator",
    "SingleBatchIterator",
    "BucketIterator",
    "HierarchicalDatasetIterator",
]

__name__ = "podium"
__version__ = "1.1.0"

# Reference for initialization of logging scikit-learn
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/__init__.py
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.WARNING)

# More information about logging can be found on project github
# https://github.com/TakeLab/podium/wiki/Logging
