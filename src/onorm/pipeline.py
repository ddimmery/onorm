from typing import List

import numpy as np

from .normalization_base import Normalizer


class Pipeline(Normalizer):
    """**Pipeline of multiple normalizers**"""

    def __init__(self, normalizers: List[Normalizer]) -> None:
        self.normalizers = normalizers

    def partial_fit(self, x: np.ndarray) -> None:
        for normalizer in self.normalizers:
            normalizer.partial_fit(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        for normalizer in self.normalizers:
            x = normalizer.transform(x)
        return x
