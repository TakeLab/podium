"""Module contains podium compatible datasets that use PyArrow as the data storage backend"""

from .arrow_tabular_dataset import ArrowDataset

__all__ = ["ArrowDataset"]
