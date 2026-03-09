"""
Random Resource Allocation Baseline
Allocates bandwidth, power, and CPU randomly to all vehicles
"""

import numpy as np
from typing import Tuple, List


class RandomAllocation:
    """Random allocation: uniform random distribution to each vehicle"""

    def __init__(self, n_vehicles: int, config):
        self.n_vehicles = n_vehicles
        self.config = config

    def allocate(self) -> np.ndarray:
        """
        Generate random resource allocation
        Returns:
            actions: shape (n_vehicles, 3) with values in [0, 1]
        """
        actions = np.random.uniform(0, 1, (self.n_vehicles, 3))
        return actions

    def get_name(self) -> str:
        return "Random"
