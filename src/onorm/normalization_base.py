from abc import ABCMeta, abstractmethod

import numpy as np


class Normalizer(metaclass=ABCMeta):
    """**Base class**

    This is the base class for all normalizers. They use a standard API.
    """

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def partial_fit(self, x: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def partial_fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.partial_fit(x)
        return self.transform(x)
