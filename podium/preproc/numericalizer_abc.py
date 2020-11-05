from abc import ABC, abstractmethod
from typing import List

import numpy as np


class NumericalizerABC(ABC):

    def __init__(self, eager=True):
        self._finalized = False
        self._eager = eager

    @abstractmethod
    def numericalize(self, tokens: List[str]) -> np.ndarray:
        pass

    def _finalize(self):
        # Subclasses should override this method to add custom
        # finalization logic
        pass

    def update(self, tokens: List[str]) -> None:
        pass

    def finalize(self) -> None:
        self._finalize()
        self._finalized = True
        pass

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def eager(self) -> bool:
        return self._eager

    def __call__(self, tokens: List[str]) -> np.ndarray:
        return self.numericalize(tokens)
