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


__name__ = "podium"
__version__ = "1.0.1"
