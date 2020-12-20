from abc import ABC, abstractmethod
from typing import List

import numpy as np


class NumericalizerABC(ABC):
    """ABC that contains the interface for Podium numericalizers. Numericalizers are used
    to transform tokens into vectors or any other custom datatype during batching.

    Attributes
    ----------
    finalized: bool
        Whether this numericalizer was finalized and is able to be used for
        numericalization.
    """

    def __init__(self, eager=True):
        """Initialises the Numericalizer.

        Parameters
        ----------
        eager: bool
            Whether the Numericalizer is to be updated during loading of the dataset, or
            after all data is loaded.

        """
        self._finalized = False
        self._eager = eager

    @abstractmethod
    def numericalize(self, tokens: List[str]) -> np.ndarray:
        """Converts `tokens` into a numericalized format used in batches.
        Numericalizations are most often numpy vectors, but any custom datatype is
        supported.

        Parameters
        ----------
        tokens: List[str]
            A list of strings that represent the tokens of this data point. Can also be
            any other datatype, as long as this Numericalizer supports it.

        Returns
        -------
        Numericalization used in batches. Numericalizations are most often numpy vectors,
        but any custom datatype is supported.
        """
        pass

    def finalize(self):
        """Finalizes the Numericalizer and prepares it for numericalization.
        This method must be overridden in classes that require finalization before
        numericalization. The override must call `mark_finalize` after successful
        completion."""
        self.mark_finalized()
        pass

    def update(self, tokens: List[str]) -> None:
        """Updates this Numericalizer with a single data point. Numericalizers that need
        to be updated example by example must override this method. Numericalizers that
        are eager get updated during the dataset loading process, while non-eager ones get
        updated after loading is finished, after all eager numericalizers were fully
        updated.

        Parameters
        ----------
        tokens: List[str]
            A list of strings that represent the tokens of this data point. Can also be
            any other datatype, as long as this Numericalizer supports it.

        """
        pass

    def mark_finalized(self) -> None:
        """Marks the field as finalized. This method must be called after finalization
        completes successfully."""
        self._finalized = True

    @property
    def finalized(self) -> bool:
        """Whether this Numericalizer was finalized and is ready for numericalization."""
        return self._finalized

    @property
    def eager(self) -> bool:
        """Whether this Numericalizer is eager. Numericalizers that
        are eager get updated during the dataset loading process, while non-eager ones get
        updated after loading is finished, after all eager numericalizers were fully
        updated."""
        return self._eager

    def __call__(self, tokens: List[str]) -> np.ndarray:
        return self.numericalize(tokens)
