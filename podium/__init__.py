"""
Home of the TakeLab Podium project. Podium is a Python machine learning library
that helps users to accelerate use of natural language processing models.

See http://takelab.fer.hr/podium/ for complete documentation.
"""
import logging

from .datasets import (
    BucketIterator,
    Dataset,
    Example,
    HierarchicalDataset,
    HierarchicalIterator,
    Iterator,
    SingleBatchIterator,
    TabularDataset,
)
from .field import Field, LabelField, MultilabelField, MultioutputField
from .vocab import Vocab


__name__ = "podium"
__version__ = "0.1.0"
