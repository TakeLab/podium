"""
Home of the TakeLab Podium project. Podium is a Python machine learning library
that helps users to accelerate use of natural language processing models.
GitHub repository: https://github.com/mtutek/podium
"""
import logging
import logging.config

from . import (
    arrow,
    dataload,
    datasets,
    metrics,
    model_selection,
    models,
    pipeline,
    preproc,
    storage,
    validation,
)

from .storage import (
    Field,
    LabelField,
    MultilabelField,
    MultioutputField,
    Vocab,
)

from .datasets import (
    Dataset,
    TabularDataset,
    HierarchicalDataset,
    Iterator,
    SingleBatchIterator,
    BucketIterator,
    HierarchicalDatasetIterator,
)

from .dataload import HuggingFaceDatasetConverter


__name__ = "podium"

__all__ = [
    "arrow",
    "dataload",
    "datasets",
    "metrics",
    "models",
    "preproc",
    "storage",
    "validation",
    "model_selection",
    "pipeline",
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
    "HuggingFaceDatasetConverter",
]


# Reference for initialization of logging scikit-learn
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/__init__.py
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.WARNING)

# More information about logging can be found on project github
# https://github.com/mtutek/podium/wiki/Logging
