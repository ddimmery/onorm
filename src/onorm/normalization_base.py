from abc import ABCMeta, abstractmethod

import numpy as np


class Normalizer(metaclass=ABCMeta):
    """Base class

    This is the base class for all normalizers. They use a standard API.
    """

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def partial_fit(self, x: np.ndarray) -> None:
        """Progressive fitting of normalization model

        This method takes a vector of data and updates the normalization
        model. The specifics of that model depend on the particular implementation.

        Args:
            x: A 1d array representing a new observation.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize data

        This method takes a vector of data and uses the normalization model to
        appropriately transform the data. The specifics of that model depend on the
        particular implementation.

        Args:
            x: A 1d array representing an observation to normalize.
        """
        raise NotImplementedError

    def partial_fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit normalization model and transform data

        This method takes a vector of data, updates the normalization model and transforms the
        supplied data. The specifics of that model depend on the particular implementation.

        Args:
            x: A 1d array representing a new observation to normalize.
        """
        self.partial_fit(x)
        return self.transform(x)

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
