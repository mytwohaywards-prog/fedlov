"""
Equal Resource Allocation Baseline
Equally divides bandwidth, power, and CPU among all vehicles
"""

import numpy as np


class EqualAllocation:
    """Equal allocation: uniform distribution to each vehicle"""

    def __init__(self, n_vehicles: int, config):
        self.n_vehicles = n_vehicles
        self.config = config
        # Pre-compute equal ratios
        self.bw_ratio = 1.0 / n_vehicles
        self.power_ratio = 1.0 / n_vehicles
        self.cpu_ratio = 1.0 / n_vehicles

    def allocate(self) -> np.ndarray:
        """
        Generate equal resource allocation
        Returns:
            actions: shape (n_vehicles, 3) with equal values
        """
        actions = np.ones((self.n_vehicles, 3)) * np.array([
            self.bw_ratio,
            self.power_ratio,
            self.cpu_ratio
        ])
        return actions

    def get_name(self) -> str:
        return "Equal"
