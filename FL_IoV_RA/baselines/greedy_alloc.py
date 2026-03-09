"""
Greedy Resource Allocation Baseline
Allocates resources greedily based on current channel conditions and queue status
"""

import numpy as np
from typing import List


class GreedyAllocation:
    """
    Greedy allocation strategy:
    1. Allocate more resources to vehicles with better channel conditions
    2. Prioritize vehicles with larger task queues
    """

    def __init__(self, n_vehicles: int, config):
        self.n_vehicles = n_vehicles
        self.config = config
        self.last_channel_gains = None

    def allocate(self, channel_gains: np.ndarray = None,
                 queue_sizes: np.ndarray = None) -> np.ndarray:
        """
        Generate greedy resource allocation
        Args:
            channel_gains: shape (n_vehicles, n_rb), if None use uniform
            queue_sizes: shape (n_vehicles,), if None use uniform

        Returns:
            actions: shape (n_vehicles, 3) with values in [0, 1]
        """
        actions = np.zeros((self.n_vehicles, 3))

        # Default: uniform distribution
        if channel_gains is None and queue_sizes is None:
            actions = np.ones((self.n_vehicles, 3)) / self.n_vehicles
            return actions

        # Compute priority scores
        if channel_gains is not None:
            # Higher channel gain -> higher priority
            avg_gains = np.mean(channel_gains, axis=1)
            gain_scores = (avg_gains - np.min(avg_gains)) / (np.max(avg_gains) - np.min(avg_gains) + 1e-10)
        else:
            gain_scores = np.ones(self.n_vehicles) / self.n_vehicles

        if queue_sizes is not None:
            # Larger queue -> higher priority
            queue_scores = (queue_sizes - np.min(queue_sizes)) / (np.max(queue_sizes) - np.min(queue_sizes) + 1e-10)
        else:
            queue_scores = np.ones(self.n_vehicles) / self.n_vehicles

        # Combined priority: weight both metrics
        priority = 0.5 * gain_scores + 0.5 * queue_scores
        priority = priority / np.sum(priority)

        # Allocate resources proportional to priority
        for i in range(self.n_vehicles):
            actions[i, 0] = priority[i]  # bandwidth
            actions[i, 1] = priority[i]  # power
            actions[i, 2] = priority[i]  # CPU

        return actions

    def get_name(self) -> str:
        return "Greedy"
