from abc import ABCMeta
from typing import Any

import numpy as np
from onorm import Normalizer


class Model(metaclass=ABCMeta):
    def __init__(self, name: str, normalizer: Normalizer, assigner: Any) -> None:
        self.normalizer = normalizer
        self.assigner = assigner
        self.name = name

    def normalize(self, x: np.ndarray) -> np.ndarray:
        self.normalizer.partial_fit(x)
        return self.normalizer.transform(x)

    def assign(self, x: np.ndarray) -> np.ndarray:
        return self.assigner.assign_next(x)
