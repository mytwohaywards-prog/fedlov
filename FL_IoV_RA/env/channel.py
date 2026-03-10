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

        # ========== 新增: 干扰信道矩阵 (公式6中的ḡ_{k',k}[m]) ==========
        # 从vehicle j到vehicle i的信道增益 (用于干扰计算)
        self.interference_channel_gain = np.ones((n_vehicles, n_vehicles, n_rb))

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
        self._compute_interference_channels()  # 新增

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
        self._compute_interference_channels()  # 新增

    def _update_path_loss(self):
        """
        Calculate path loss based on distance using 3GPP model
        严格遵循论文Table II的双层路径损耗模型

        V2V Link (LOS/NLOS):
        LOS:  PL(dB) = 38.77 + 16.7*log10(d) + 18.2*log10(fc)
        NLOS: PL(dB) = 36.85 + 30*log10(d) + 18.9*log10(fc)

        其中 d 为距离(meters), fc 为频率(GHz)
        """
        fc_GHz = self.config.CARRIER_FREQUENCY / 1e9  # 转换为GHz (4.7 GHz)

        # Calculate distances (all pairs)
        for i in range(self.n_vehicles):
            for j in range(self.n_vehicles):
                if i != j:
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    dist = max(dist, self.config.MIN_DISTANCE)
                else:
                    # Self-link to RSU
                    dist = 100.0

                # 判断LOS/NLOS状态
                # 论文: V2V同一条道路上时为LOS，概率为plos(d) = ... (简化：距离小于50m为LOS)
                is_los = dist < 50.0

                # 应用3GPP路径损耗公式 (论文Table II)
                if is_los:
                    # LOS model
                    path_loss_db = 38.77 + 16.7 * np.log10(dist) + 18.2 * np.log10(fc_GHz)
                else:
                    # NLOS model
                    path_loss_db = 36.85 + 30 * np.log10(dist) + 18.9 * np.log10(fc_GHz)

                # Apply to all RBs (path loss不随RB变化)
                for rb in range(self.n_rb):
                    self.path_loss[i, rb] = path_loss_db

    def _update_shadowing_initial(self):
        """
        Initialize shadowing with proper LOS/NLOS standards
        论文Table II: LOS下σ=3dB, NLOS下σ=4dB
        """
        # 先计算LOS/NLOS状态（与_update_path_loss中的逻辑一致）
        los_matrix = np.zeros((self.n_vehicles, self.n_rb), dtype=bool)
        for i in range(self.n_vehicles):
            for j in range(self.n_vehicles):
                if i != j:
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    dist = max(dist, self.config.MIN_DISTANCE)
                    los_matrix[i, :] = (dist < 50.0)

        # 根据LOS/NLOS应用不同的标准差
        self.shadowing_state = np.zeros((self.n_vehicles, self.n_rb))
        for i in range(self.n_vehicles):
            for rb in range(self.n_rb):
                if los_matrix[i, rb]:
                    # LOS: σ = 3 dB
                    self.shadowing_state[i, rb] = np.random.normal(0, self.config.SHADOWING_STD)
                else:
                    # NLOS: σ = 4 dB
                    self.shadowing_state[i, rb] = np.random.normal(0, self.config.SHADOWING_STD_NLOS)

        self.shadowing = self.shadowing_state.copy()

    def _update_shadowing_smooth(self):
        """
        Update shadowing with exponential correlation
        论文Table II: 每100ms更新一次 (SLOT_DURATION=100ms)
        当前在每个step调用，实现smooth fading
        """
        correlation_coef = 0.9  # correlation between slots

        # 计算当前LOS/NLOS状态
        los_matrix = np.zeros((self.n_vehicles, self.n_rb), dtype=bool)
        for i in range(self.n_vehicles):
            for j in range(self.n_vehicles):
                if i != j:
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    dist = max(dist, self.config.MIN_DISTANCE)
                    los_matrix[i, :] = (dist < 50.0)

        # 根据LOS/NLOS选择不同的标准差进行更新
        new_shadowing = np.zeros((self.n_vehicles, self.n_rb))
        for i in range(self.n_vehicles):
            for rb in range(self.n_rb):
                if los_matrix[i, rb]:
                    shadowing_std = self.config.SHADOWING_STD  # 3 dB for LOS
                else:
                    shadowing_std = self.config.SHADOWING_STD_NLOS  # 4 dB for NLOS

                new_shadowing[i, rb] = (correlation_coef * self.shadowing_state[i, rb] +
                                       (1 - correlation_coef) * np.random.normal(0, shadowing_std))
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

    def _compute_interference_channels(self):
        """
        计算干扰信道增益矩阵 (公式6中的ḡ_{k',k}[m])

        从vehicle j到vehicle i的信道增益，用于计算vehicle i收到的干扰

        对于每对(i, j)，计算基于i-j之间的距离的路径损耗
        """
        fc_GHz = self.config.CARRIER_FREQUENCY / 1e9

        for i in range(self.n_vehicles):
            for j in range(self.n_vehicles):
                if i == j:
                    # 自环：vehicle自己到自己的信道
                    for rb in range(self.n_rb):
                        self.interference_channel_gain[j, i, rb] = self.channel_gain[i, rb]
                else:
                    # j到i的干扰信道
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    dist = max(dist, self.config.MIN_DISTANCE)

                    # 判断LOS/NLOS
                    is_los = dist < 50.0

                    # 路径损耗
                    if is_los:
                        path_loss_db = 38.77 + 16.7 * np.log10(dist) + 18.2 * np.log10(fc_GHz)
                    else:
                        path_loss_db = 36.85 + 30 * np.log10(dist) + 18.9 * np.log10(fc_GHz)

                    # 将路径损耗转换为线性
                    pl_linear = 10 ** (-path_loss_db / 10)

                    # 阴影衰落：对于j到i的链路，使用近似阴影值
                    # (简化：假设与i的本地阴影相同)
                    shadowing_linear = 10 ** (-self.shadowing[i] / 10)

                    # 快衰落：对于干扰链路，生成新的独立衰落
                    # (为了简化，使用与main channel相同的fading)
                    fading_power = self.fading[j] ** 2

                    # 完整干扰信道增益
                    for rb in range(self.n_rb):
                        self.interference_channel_gain[j, i, rb] = pl_linear * shadowing_linear[rb] * fading_power[rb]

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
        r"""
        Compute SINR for all vehicles
        严格按照论文公式(5)和(6)实现

        公式(5): γ_k^v[m] = P_k^v[m]g_k[m] / (I_k[m] + σ²)
        公式(6): I_k[m] = P_m^c[m]ḡ_{m,k}[m] + Σ_{k'∈K\k} ρ_{k'}[m]P_{k'}^v[m]ḡ_{k',k}[m]

        Args:
            power_linear: shape (n_vehicles,) in linear scale (Watts)
            rb_indices: shape (n_vehicles,) RB indices selected by each vehicle (1-indexed: 1 to N_RB)

        Returns:
            sinr: shape (n_vehicles,) in linear scale
            sinr_db: shape (n_vehicles,) in dB
        """
        noise_power = 10 ** (self.config.NOISE_POWER_dBm / 10) * 1e-3  # Convert dBm to W

        sinr = np.zeros(self.n_vehicles)

        # Compute received signal power and interference
        for i in range(self.n_vehicles):
            rb_idx = int(rb_indices[i])

            # 期望信号功率: P_k^v[m] * g_k[m]
            # rb_idx范围: 1到N_RB，所以数组索引为rb_idx-1
            if rb_idx >= 1 and rb_idx <= self.n_rb:
                # 使用正确的RB
                rb_array_idx = rb_idx - 1
                desired_power = power_linear[i] * self.channel_gain[i, rb_array_idx]
            else:
                # 无效RB，使用平均信道增益
                desired_power = power_linear[i] * np.mean(self.channel_gain[i])
                rb_array_idx = 0

            # 干扰功率: 公式(6)
            # I_k[m] = Σ_{k'∈K\k} ρ_{k'}[m]P_{k'}^v[m]ḡ_{k',k}[m]
            # 其中ḡ_{k',k}[m] = interference_channel_gain[k', i, m] (从k'到i的信道增益)
            interference_power = 0
            if rb_idx >= 1 and rb_idx <= self.n_rb:
                for j in range(self.n_vehicles):
                    if i != j:
                        rb_idx_j = int(rb_indices[j])
                        if rb_idx == rb_idx_j:
                            # Vehicle j在同一RB上发射，对vehicle i产生干扰
                            # 使用干扰信道增益矩阵而非自身信道
                            interference_power += power_linear[j] * self.interference_channel_gain[j, i, rb_array_idx]

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
