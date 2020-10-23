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
    TokenizedField,
    unpack_fields,
    Vocab
)

from .datasets import (
    Dataset,
    TabularDataset,
    HierarchicalDataset,
    Iterator,
    SingleBatchIterator,
    BucketIterator,
    HierarchicalDatasetIterator
)


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
    "Field",
    "LabelField",
    "MultilabelField",
    "MultioutputField",
    "TokenizedField",
    "unpack_fields",
    "Vocab",
    "Dataset",
    "TabularDataset",
    "HierarchicalDataset",
    "Iterator",
    "SingleBatchIterator",
    "BucketIterator",
    "HierarchicalDatasetIterator"
]


# Reference for initialization of logging scikit-learn
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/__init__.py
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.WARNING)

# More information about logging can be found on project github
# https://github.com/mtutek/podium/wiki/Logging
