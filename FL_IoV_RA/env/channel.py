"""
V2V/V2I Channel Model
Reference: Environment_Platoon.py from V2X-RRM-IEEE-OJ-COMS-2024-main
Includes: Path Loss, Log-Normal Shadowing, and Rayleigh/Nakagami Fading
"""

import numpy as np
import math
from typing import Tuple


class ChannelModel:
    """Vehicle-to-Vehicle / Vehicle-to-Infrastructure Channel Model"""

    def __init__(self, n_vehicles: int, n_rb: int, config):
        self.n_vehicles = n_vehicles
        self.n_rb = n_rb
        self.config = config

        # Channel state matrices
        self.path_loss = np.zeros((n_vehicles, n_rb))  # dB
        self.shadowing = np.zeros((n_vehicles, n_rb))  # dB
        self.fading = np.ones((n_vehicles, n_rb))      # linear scale
        self.channel_gain = np.ones((n_vehicles, n_rb)) # linear scale

        # Shadowing correlation state for smoothness
        self.shadowing_state = np.zeros((n_vehicles, n_rb))

        # Interference matrix
        self.interference = np.zeros((n_vehicles, n_rb))

        # Positions
        self.positions = None  # Will be set by environment

    def reset(self, positions: np.ndarray):
        """
        Reset channel with initial positions
        Args:
            positions: shape (n_vehicles, 2) [x, y] coordinates
        """
        self.positions = positions.copy()
        self._update_path_loss()
        self._update_shadowing_initial()
        self._update_fading()
        self._compute_channel_gains()

    def update(self, positions: np.ndarray):
        """
        Update channel for new vehicle positions (mobility)
        Args:
            positions: shape (n_vehicles, 2)
        """
        self.positions = positions.copy()
        self._update_path_loss()
        self._update_shadowing_smooth()
        self._update_fading()
        self._compute_channel_gains()

    def _update_path_loss(self):
        """
        Calculate path loss based on distance
        Model: PL(d) = PL_0 + 10*n*log10(d/d_0)
        where PL_0 = reference path loss at 1m, n = path loss exponent
        """
        d0 = 1.0  # reference distance in meters
        # Reference path loss at 1m: PL(1m) = 20*log10(f) + 20*log10(4π/c) - 20
        f_GHz = self.config.CARRIER_FREQUENCY / 1e9
        PL_0_dB = 20 * np.log10(f_GHz) + 20 * np.log10(4 * np.pi / 3e8) - 20

        # Calculate distances (all pairs)
        for i in range(self.n_vehicles):
            for j in range(self.n_vehicles):
                if i != j:
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    dist = max(dist, self.config.MIN_DISTANCE)
                else:
                    # Self-interference from base station/RSU
                    dist = 100  # fixed distance to RSU

                path_loss_db = PL_0_dB + 10 * self.config.PATH_LOSS_EXPONENT * np.log10(dist / d0)

                # Apply to all RBs (same path loss per RB in this simplified model)
                for rb in range(self.n_rb):
                    self.path_loss[i, rb] = path_loss_db

    def _update_shadowing_initial(self):
        """Initialize shadowing with random values"""
        self.shadowing_state = np.random.normal(0, self.config.SHADOWING_STD,
                                               (self.n_vehicles, self.n_rb))
        self.shadowing = self.shadowing_state.copy()

    def _update_shadowing_smooth(self):
        """
        Update shadowing with exponential correlation
        Models the slow variation of shadowing across time slots
        """
        correlation_coef = 0.9  # correlation between slots
        new_shadowing = (correlation_coef * self.shadowing_state +
                        (1 - correlation_coef) * np.random.normal(0, self.config.SHADOWING_STD,
                                                                   (self.n_vehicles, self.n_rb)))
        self.shadowing_state = new_shadowing
        self.shadowing = self.shadowing_state.copy()

    def _update_fading(self):
        """
        Generate fast fading coefficients
        Rayleigh fading: |h| ~ Rayleigh, |h|^2 ~ Exponential(1)
        Nakagami fading: shape parameter m
        """
        if self.config.FADING_MODEL == "rayleigh":
            # Rayleigh: sqrt of two independent Gaussians
            self.fading = np.sqrt(np.random.exponential(1, (self.n_vehicles, self.n_rb)))
        elif self.config.FADING_MODEL == "nakagami":
            m = self.config.NAKAGAMI_M
            # Nakagami: shape=m, scale=1
            self.fading = np.sqrt(np.random.gamma(m, 1/m, (self.n_vehicles, self.n_rb)))
        else:
            # No fading
            self.fading = np.ones((self.n_vehicles, self.n_rb))

    def _compute_channel_gains(self):
        """
        Compute composite channel gain = |h|^2
        G = 10^(-PL/10) * 10^(-Shadowing/10) * |h_fading|^2
        """
        PL_linear = 10 ** (-self.path_loss / 10)
        shadowing_linear = 10 ** (-self.shadowing / 10)
        fading_power = self.fading ** 2

        self.channel_gain = PL_linear * shadowing_linear * fading_power

    def get_channel_gain(self, vehicle_idx: int = None) -> np.ndarray:
        """
        Get channel gain for vehicles
        Returns:
            If vehicle_idx is None: shape (n_vehicles, n_rb)
            If vehicle_idx is given: shape (n_rb,)
        """
        if vehicle_idx is None:
            return self.channel_gain.copy()
        else:
            return self.channel_gain[vehicle_idx].copy()

    def compute_sinr(self, power_linear: np.ndarray,
                    rb_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SINR for all vehicles
        Args:
            power_linear: shape (n_vehicles,) in linear scale (Watts)
            rb_indices: shape (n_vehicles,) RB indices selected by each vehicle

        Returns:
            sinr: shape (n_vehicles,) in linear scale
            sinr_db: shape (n_vehicles,) in dB
        """
        noise_power = 10 ** (self.config.NOISE_POWER_dBm / 10) * 1e-3  # Convert dBm to W

        sinr = np.zeros(self.n_vehicles)

        # Compute received signal power and interference
        for i in range(self.n_vehicles):
            rb_idx = int(rb_indices[i])

            # Desired signal on selected RB
            if rb_idx > 0 and rb_idx <= self.n_rb:
                desired_power = power_linear[i] * self.channel_gain[i, rb_idx - 1]  # RB索引从1开始
            else:
                desired_power = power_linear[i] * np.mean(self.channel_gain[i])  # 无效RB，使用平均

            # Interference from other vehicles on same RB
            interference_power = 0
            for j in range(self.n_vehicles):
                if i != j:
                    rb_idx_j = int(rb_indices[j])
                    if rb_idx == rb_idx_j and rb_idx > 0:
                        # Same RB: direct interference
                        interference_power += power_linear[j] * self.channel_gain[j, rb_idx - 1]

            total_interference = interference_power + noise_power
            sinr[i] = desired_power / (total_interference + 1e-10)  # avoid division by zero

        sinr_db = 10 * np.log10(sinr + 1e-10)

        return sinr, sinr_db

    def compute_rate(self, sinr: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        """
        Compute transmission rate using Shannon capacity
        Rate = B * log2(1 + SINR)

        Args:
            sinr: shape (n_vehicles,) in linear scale
            bandwidth: shape (n_vehicles,) in Hz

        Returns:
            rate: shape (n_vehicles,) in bits per second
        """
        rate = bandwidth * np.log2(sinr + 1)
        return rate


class InterferenceCoordinator:
    """Manage interference between vehicles (simplified)"""

    def __init__(self, n_vehicles: int, n_rb: int):
        self.n_vehicles = n_vehicles
        self.n_rb = n_rb
        self.interference_matrix = np.zeros((n_vehicles, n_rb))

    def estimate_interference(self, channel: ChannelModel,
                            power_allocation: np.ndarray) -> np.ndarray:
        """
        Estimate interference on each RB
        Args:
            channel: ChannelModel instance
            power_allocation: shape (n_vehicles, n_rb) power per vehicle per RB

        Returns:
            interference: shape (n_vehicles, n_rb) interference power seen
        """
        interference = np.zeros((self.n_vehicles, self.n_rb))

        for i in range(self.n_vehicles):
            for rb in range(self.n_rb):
                for j in range(self.n_vehicles):
                    if i != j:
                        # Interference from vehicle j on vehicle i at RB
                        tx_power = power_allocation[j, rb]
                        channel_gain = channel.get_channel_gain(j)[rb]
                        interference[i, rb] += tx_power * channel_gain

        self.interference_matrix = interference.copy()
        return interference
