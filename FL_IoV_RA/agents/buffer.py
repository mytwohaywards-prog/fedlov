"""
Experience Replay Buffer for RL agents
Reference: buffer.py from FedCola project
"""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-size experience replay buffer for storing transitions
    Supports sampling for mini-batch training
    """

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int):
        """
        Initialize replay buffer
        Args:
            buffer_size: maximum capacity of the buffer
            obs_dim: dimension of observation space
            act_dim: dimension of action space
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Pre-allocate storage
        self.obs_buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs_buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0  # pointer to current position
        self.size = 0  # actual size of buffer

    def store_transition(self, obs: np.ndarray, action: np.ndarray,
                         reward: float, next_obs: np.ndarray, done: bool):
        """
        Store a transition in the buffer
        Args:
            obs: observation (obs_dim,)
            action: action (act_dim,)
            reward: scalar reward
            next_obs: next observation (obs_dim,)
            done: episode termination flag
        """
        self.obs_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random mini-batch from the buffer
        Args:
            batch_size: size of the mini-batch

        Returns:
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
            Each with shape (batch_size, ...)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        obs_batch = self.obs_buffer[indices]
        action_batch = self.action_buffer[indices]
        reward_batch = self.reward_buffer[indices]
        next_obs_batch = self.next_obs_buffer[indices]
        done_batch = self.done_buffer[indices]

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def is_ready(self, batch_size: int = None) -> bool:
        """
        Check if buffer has enough samples for training
        Args:
            batch_size: required minimum size (default: use buffer_size/10)
        Returns:
            True if buffer.size >= batch_size
        """
        if batch_size is None:
            batch_size = max(32, self.buffer_size // 100)
        return self.size >= batch_size

    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0

    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            'buffer_size': self.size,
            'capacity': self.buffer_size,
            'fill_ratio': self.size / self.buffer_size,
        }

    def __len__(self):
        return self.size
