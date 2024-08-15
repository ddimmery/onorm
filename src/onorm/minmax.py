import numpy as np

from .normalization_base import Normalizer


class MinMaxScaler(Normalizer):
    """Min-Max Scaling for each feature

    The class performs element-wise online updates of a minimum and maximum and
    then normalizes so that each element, $i$, of the vector $x_t$ is guaranteed to lie
    within zero and one.

    Define the minimum and maximum at time $t$:

    $$\\textrm{mn}_{ti} = \\min \\{x_{1i},\\dots x_{ti}\\}$$

    $$\\textrm{mx}_{ti} = \\max \\{x_{1i},\\dots x_{ti}\\}$$

    Then normalization is:

    $$\\frac{x_{ti} - \\textrm{mn}_{ti}}{\\textrm{mx}_{ti} - \\textrm{mn}_{ti}}$$
    """

    def __init__(self, n_dim: int) -> None:
        self.min = np.array([np.inf] * n_dim)
        self.max = np.array([-np.inf] * n_dim)

    def _update_min(self, x: np.ndarray) -> None:
        self.min = np.fmin(self.min, x)

    def _update_max(self, x: np.ndarray) -> None:
        self.max = np.fmax(self.max, x)

    def partial_fit(self, x: np.ndarray) -> None:
        """Update the minimum and maximum for each feature.

        Compares the new feature vector to the previous minimum
        and maximum for each dimension and updates the state.

        Args:
            x: A 1d array representing a new observation.
        """
        self._update_min(x)
        self._update_max(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the feature vector according to the current state

        If the current minimum and maximum are equal, then the transformation
        returns $x_{ti} - \\textrm{mn}_{ti}$.

        Args:
            x: A 1d array representing an observation to normalize.
        """
        denom = self.max - self.min
        if np.linalg.norm(denom) <= np.finfo(np.float64).eps:
            denom = 1
        return (x - self.min) / denom
