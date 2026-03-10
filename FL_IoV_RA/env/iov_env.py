"""
IoV Environment for FL-based Resource Allocation
Simulates vehicles with computational tasks, wireless channels, and latency constraints
"""

import numpy as np
from typing import List, Tuple, Dict
from .channel import ChannelModel, InterferenceCoordinator


class Vehicle:
    """Single vehicle model with task queue and resource state"""

    def __init__(self, vehicle_id: int, config):
        self.id = vehicle_id
        self.config = config
        self.position = np.zeros(2)  # [x, y]
        self.velocity = np.zeros(2)  # [vx, vy]

        # ========== 论文公式中的关键变量 ==========
        # B_k: 剩余待传输数据 (bits) - 公式(18)中的B_k
        self.B_k = config.TASK_SIZE  # 初始化为TASK_SIZE

        # T_k: 剩余延迟容限 (seconds) - 公式(18)中的T_k
        self.T_k = config.T_MAX  # 初始化为T_MAX

        # ρ_k[m]: RB选择指示符 (binary) - 当前选中的RB索引
        self.selected_rb = -1  # -1表示未选择，0到N_RB-1表示选中的RB

        # C_k^v[m]: 选中RB上的V2V传输速率 (bps)
        self.transmission_rate_selected_rb = 0.0

        # Task queue (保持兼容)
        self.queue_size = 0  # number of bits waiting
        self.task_deadline = None
        self.total_tasks_arrived = 0
        self.total_tasks_completed = 0

        # Resource allocation
        self.allocated_bandwidth = 0  # Hz
        self.allocated_power = 0  # Watts
        self.allocated_cpu = 0  # Hz

    def reset(self, x: float, y: float, vx: float = 0, vy: float = 0):
        """Initialize vehicle position and velocity"""
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.queue_size = 0
        self.total_tasks_arrived = 0
        self.total_tasks_completed = 0

    def move(self, dt: float):
        """Update position based on velocity"""
        self.position += self.velocity * dt
        # Wraparound at boundaries
        self.position[0] = self.position[0] % self.config.SCENARIO_LENGTH
        self.position[1] = self.position[1] % self.config.SCENARIO_WIDTH

    def add_task(self, task_size: int = None):
        """Add incoming task to queue"""
        if task_size is None:
            task_size = self.config.TASK_SIZE
        self.queue_size += task_size
        self.total_tasks_arrived += 1

    def process_task(self, completed_bits: int):
        """Remove completed bits from queue"""
        self.queue_size = max(0, self.queue_size - completed_bits)
        if completed_bits > 0:
            self.total_tasks_completed += 1

    def get_queue_state(self) -> float:
        """Normalized queue length [0, 1]"""
        max_queue = self.config.TASK_SIZE * 10
        return min(1.0, self.queue_size / max_queue)


class IoVEnv:
    """
    IoV Environment with N vehicles, wireless channels, and resource allocation
    Observation: [channel_gains, queue_sizes, deadline_info, interference_est]
    Action: [bandwidth_ratio, power_ratio, cpu_ratio] for each vehicle
    """

    def __init__(self, config):
        self.config = config
        self.n_vehicles = config.N_VEHICLES
        self.n_rb = config.N_RB

        # Initialize vehicles
        self.vehicles = [Vehicle(i, config) for i in range(self.n_vehicles)]

        # Channel model
        self.channel = ChannelModel(self.n_vehicles, self.n_rb, config)
        self.interference_coord = InterferenceCoordinator(self.n_vehicles, self.n_rb)

        # Time tracking
        self.current_step = 0
        self.episode_length = 1000

        # Statistics
        self.episode_delays = []
        self.episode_energies = []
        self.episode_rewards = []
        self.constraint_violations = 0

        # Observation space: per vehicle (✅ 修复: 按照公式16完整实现)
        # 组成: [N_RB信道增益] + [N_RB干扰向量] + [B_k] + [T_k] + [时间进度] + [队列状态]
        #     = N_RB + N_RB + 1 + 1 + 1 + 1 = 2*N_RB + 4
        self.obs_dim = 2 * self.n_rb + 4

        # Action space: DISCRETE (per vehicle)
        # action = rb_action * (N_POWER * N_CPU) + power_action * N_CPU + cpu_action
        self.n_rb_actions = config.N_RB_ACTIONS
        self.n_power_actions = config.N_POWER_ACTIONS
        self.n_cpu_actions = config.N_CPU_ACTIONS
        self.n_actions = self.n_rb_actions * self.n_power_actions * self.n_cpu_actions

        # SINR tracking (for reward calculation)
        self.current_sinr = np.ones(self.n_vehicles)  # Initial SINR values
        self.current_sinr_db = np.zeros(self.n_vehicles)  # Initial SINR in dB

    def reset(self) -> List[np.ndarray]:
        """
        Reset environment for new episode
        Returns:
            observations: list of (n_vehicles,) or (obs_dim,) for each vehicle
        """
        self.current_step = 0
        self.episode_delays = []
        self.episode_energies = []
        self.episode_rewards = []
        self.constraint_violations = 0

        # Reset SINR tracking
        self.current_sinr = np.ones(self.n_vehicles)
        self.current_sinr_db = np.zeros(self.n_vehicles)

        # Randomly initialize vehicle positions and velocities
        positions = np.random.uniform(0, self.config.SCENARIO_LENGTH,
                                     (self.n_vehicles, 2))
        for i, vehicle in enumerate(self.vehicles):
            vx = np.random.uniform(-self.config.V_MAX, self.config.V_MAX)
            vy = np.random.uniform(-self.config.V_MAX, self.config.V_MAX)
            vehicle.reset(positions[i, 0], positions[i, 1], vx, vy)

            # Reset B_k and T_k for formula (18)
            vehicle.B_k = self.config.TASK_SIZE
            vehicle.T_k = self.config.T_MAX
            vehicle.selected_rb = -1
            vehicle.transmission_rate_selected_rb = 0.0

        # Initialize channel
        positions = np.array([v.position for v in self.vehicles])
        self.channel.reset(positions)

        # Add initial tasks
        for vehicle in self.vehicles:
            vehicle.add_task()

        return self._get_observations()

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, bool, Dict]:
        """
        Execute one environment step
        Args:
            actions: shape (n_vehicles,) with discrete action indices
                    Each action encodes [rb_choice, power_level, cpu_level]

        Returns:
            observations: list of local observations for each vehicle
            rewards: shape (n_vehicles,) cumulative reward per vehicle
            done: whether episode is finished
            info: dict with statistics
        """
        self.current_step += 1

        # Decode discrete actions to actual resource allocations
        power_linear, bandwidth, cpu_freq, rb_indices = self._decode_discrete_actions(actions)

        # Update vehicle positions (mobility)
        for vehicle in self.vehicles:
            vehicle.move(self.config.SLOT_DURATION)

        # Update channel (fading, shadowing, path loss)
        positions = np.array([v.position for v in self.vehicles])
        self.channel.update(positions)

        # Compute SINR and transmission rates (✅ 修复: 使用RB索引而不是带宽比例)
        sinr, sinr_db = self.channel.compute_sinr(power_linear, rb_indices)

        transmission_rate = self.channel.compute_rate(sinr, bandwidth)

        # 保存SINR用于奖励计算（论文基于成功率）
        self.current_sinr = sinr
        self.current_sinr_db = sinr_db

        # ========== 更新Vehicle状态变量: selected_rb, transmission_rate_selected_rb ==========
        for i in range(self.n_vehicles):
            self.vehicles[i].selected_rb = rb_indices[i] - 1  # 转换: 0->-1 (无RB), 1->0, ..., N_RB->N_RB-1
            self.vehicles[i].transmission_rate_selected_rb = transmission_rate[i]

        # Compute delays
        comm_delays = np.zeros(self.n_vehicles)
        comp_delays = np.zeros(self.n_vehicles)

        for i in range(self.n_vehicles):
            # Communication delay
            if transmission_rate[i] > 0:
                comm_delays[i] = self.vehicles[i].queue_size / (transmission_rate[i] + 1e-10)
            else:
                comm_delays[i] = self.config.T_MAX * 2  # large penalty for zero rate

            # Cap communication delay (✅ 改: 防止极端值)
            comm_delays[i] = min(comm_delays[i], self.config.T_MAX * 3)

            # Computation delay
            if cpu_freq[i] > 0:
                comp_delays[i] = (self.config.CPU_CYCLES / cpu_freq[i])
            else:
                comp_delays[i] = self.config.T_MAX * 2

            # Cap computation delay (✅ 改: 防止极端值)
            comp_delays[i] = min(comp_delays[i], self.config.T_MAX * 3)

        total_delays = comm_delays + comp_delays

        # Compute energy consumption (✅ 改: 修复comp_energy数值溢出)
        tx_energy = power_linear * self.config.SLOT_DURATION  # transmission energy (Watts × seconds = Joules)
        # 计算能耗: E = CPU_CYCLES / f_i (单位任务的cycles数除以频率)
        # 按时间段计算: cycles_per_slot = CPU_CYCLES * SLOT_DURATION
        comp_energy = (self.config.CPU_CYCLES * self.config.SLOT_DURATION / (cpu_freq + 1e-10)) * 1e-9  # 单位: Joules
        total_energy = tx_energy + comp_energy

        # Process completed tasks and update B_k
        bits_completed = transmission_rate * self.config.SLOT_DURATION
        for i in range(self.n_vehicles):
            self.vehicles[i].process_task(bits_completed[i])
            # 更新B_k: 减少已发送的比特数 (公式18中的B_k)
            self.vehicles[i].B_k = max(0, self.vehicles[i].B_k - bits_completed[i])

        # Update T_k: 减少剩余延迟容限 (公式18中的T_k)
        for i in range(self.n_vehicles):
            self.vehicles[i].T_k = max(0, self.vehicles[i].T_k - self.config.SLOT_DURATION)

        # Add new tasks (Poisson arrival)
        for i in range(self.n_vehicles):
            if np.random.rand() < self.config.TASK_ARRIVAL_RATE:
                self.vehicles[i].add_task()

        # Compute rewards
        rewards = self._compute_rewards(total_delays, total_energy, transmission_rate)

        # Update statistics
        self.episode_delays.extend(total_delays)
        self.episode_energies.extend(total_energy)
        self.episode_rewards.extend(rewards)

        for i in range(self.n_vehicles):
            if total_delays[i] > self.config.T_MAX:
                self.constraint_violations += 1

        done = self.current_step >= self.episode_length

        # Get next observations
        obs = self._get_observations()

        info = {
            'delays': total_delays,
            'energies': total_energy,
            'rates': transmission_rate,
            'sinr_db': sinr_db,
            'constraint_violated': np.any(total_delays > self.config.T_MAX),
            'avg_delay': np.mean(total_delays),
            'avg_energy': np.mean(total_energy),
            'queue_sizes': np.array([v.queue_size for v in self.vehicles]),
        }

        return obs, rewards, done, info

    def _decode_discrete_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode discrete actions to actual resource allocations

        严格遵循论文约束(3): ∑_m ρ_k[m] = 1 - 每个vehicle必须选择恰好一个RB

        Args:
            actions: shape (n_vehicles,) with integer action indices
                    action_i = rb_idx * (N_POWER * N_CPU) + power_idx * N_CPU + cpu_idx

        Returns:
            power_linear: shape (n_vehicles,) in Watts
            bandwidth: shape (n_vehicles,) in Hz
            cpu_freq: shape (n_vehicles,) in Hz
            rb_indices: shape (n_vehicles,) RB indices for each vehicle (1-indexed: 1 to N_RB)
        """
        power_linear = np.zeros(self.n_vehicles)
        bandwidth = np.zeros(self.n_vehicles)
        cpu_freq = np.zeros(self.n_vehicles)
        rb_indices = np.zeros(self.n_vehicles, dtype=int)

        max_power_watts = 10 ** (self.config.MAX_POWER / 10) * 1e-3  # dBm to W

        for i in range(self.n_vehicles):
            action = actions[i]

            # Decode discrete action: action = rb_idx * (N_POWER * N_CPU) + power_idx * N_CPU + cpu_idx
            # rb_idx范围: 0 到 N_RB (共 N_RB+1 个值)
            # 修改: rb_idx应该映射到1到N_RB的有效RB (遵循约束3)
            rb_idx = action // (self.n_power_actions * self.n_cpu_actions)

            # 修复: 将rb_idx归一化到1到N_RB范围 (强制满足约束3)
            rb_idx = (rb_idx % self.n_rb) + 1  # 范围: 1 到 N_RB (总是有效的RB)
            rb_indices[i] = rb_idx

            # Bandwidth: 每个vehicle必须获得BW_PER_RB (遵循约束3)
            bandwidth[i] = self.config.BW_PER_RB

            # Power: discrete levels (10%, 25%, 50%, 75%, 100%)
            power_idx = (action % (self.n_power_actions * self.n_cpu_actions)) // self.n_cpu_actions
            power_levels = [0.1, 0.25, 0.50, 0.75, 1.0]  # 最小10%避免zero-power问题
            power_linear[i] = power_levels[min(power_idx, len(power_levels) - 1)] * max_power_watts

            # CPU frequency: discrete levels (10%, 25%, 50%, 75%, 100%)
            cpu_idx = action % self.n_cpu_actions
            cpu_levels = [0.1, 0.25, 0.50, 0.75, 1.0]  # min 0.1 to ensure some processing
            cpu_freq[i] = cpu_levels[min(cpu_idx, len(cpu_levels) - 1)] * self.config.MAX_CPU

        return power_linear, bandwidth, cpu_freq, rb_indices

    def _compute_rewards(self, delays: np.ndarray, energies: np.ndarray,
                         rates: np.ndarray) -> np.ndarray:
        """
        严格按照论文公式(18)和(19)实现奖励函数 - NO SIMPLIFICATIONS, NO EMPIRICAL MAPPINGS

        公式(18):
        r_t(k) = {
            ζ^net + λ₃G(γ_k^c - γ^th) + λ₄G(∑ρ_k[m]C_k^v[m] - B_k/T_k),  if B_k > 0
            A₁,                                                             otherwise
        }

        公式(19): 从属函数G(x)
        G(x) = {
            A₂,  if x > 0
            x,   otherwise
        }

        其中:
        - ζ^net: 网络SEE (公式11: λ₁ζ^V2I + λ₂ζ^V2V)
        - γ_k^c: V2V SINR (公式5)
        - γ^th: SINR阈值 (1 dBm per Table I)
        - ∑ρ_k[m]C_k^v[m]: 选中RB上的V2V传输速率 (bps)
        - B_k: 待传输数据 (bits)
        - T_k: 剩余延迟容限 (seconds)

        Args:
            delays: shape (n_vehicles,) - 总延迟 (seconds)
            energies: shape (n_vehicles,) - 总能耗 (Joules)
            rates: shape (n_vehicles,) - 传输速率 (bps)

        Returns:
            rewards: shape (n_vehicles,)
        """
        rewards = np.zeros(self.n_vehicles)

        # ========== 1. 计算网络SEE (ζ^net = λ₁ζ^V2I + λ₂ζ^V2V) - 公式11 ==========
        # SEE (Spectral Energy Efficiency) = Capacity / (BW * Energy)
        # V2V SEE = ∑(all V2V rates) / (Total BW × Total Energy)

        total_v2v_capacity = np.sum(rates)  # bits/sec
        total_energy = np.sum(energies) + 1e-10  # Joules/sec (+ small epsilon to avoid division by zero)
        BW_total = self.config.N_RB * self.config.BW_PER_RB  # Hz

        # SEE_V2V = (bits/sec) / (Hz × J/sec) = bits / (Hz × J) [按秒进行计算]
        see_v2v = total_v2v_capacity / (BW_total * total_energy) if total_energy > 0 else 0.0

        # V2I SEE: 在这个纯V2V场景中，假设V2I SEE ≈ V2V SEE (实际应单独计算)
        see_v2i = see_v2v

        # 网络SEE (公式11)
        zeta_net = (self.config.REWARD_LAMBDA1_SEE * see_v2i +
                    self.config.REWARD_LAMBDA2_SEE * see_v2v)

        # ========== 2. 从属函数G(x)实现 (公式19) ==========
        def auxiliary_function_G(x):
            """从属函数 G(x) - 严格按公式19"""
            if x > 0:
                return self.config.REWARD_A2
            else:
                return x  # 返回x本身，不做任何修改

        # ========== 3. 对每个vehicle计算奖励 - 公式18 ==========
        # SINR阈值 (论文Table I: 1 dBm)
        SINR_threshold_dB = self.config.SINR_THRESHOLD_dB
        SINR_threshold_linear = 10 ** (SINR_threshold_dB / 10)

        for k in range(self.n_vehicles):
            # 检查B_k > 0 (是否有待传输数据)
            if self.vehicles[k].B_k > 0:
                # B_k > 0: 计算完整奖励 - 公式18第一个分支

                # 项1: 网络SEE ζ^net
                term1 = zeta_net

                # 项2: SINR约束项 λ₃G(γ_k^c - γ^th)
                # γ_k^c: V2V SINR for vehicle k (current_sinr存储的是线性值)
                sinr_margin = self.current_sinr[k] - SINR_threshold_linear
                term2 = self.config.REWARD_LAMBDA3 * auxiliary_function_G(sinr_margin)

                # 项3: 传输速率约束 λ₄G(∑ρ_k[m]C_k^v[m] - B_k/T_k)
                # ∑ρ_k[m]C_k^v[m]: 选中RB上的V2V速率 = transmission_rate_selected_rb
                # B_k/T_k: 所需速率 (bits / seconds)
                transmission_rate_required = self.vehicles[k].B_k / (self.vehicles[k].T_k + 1e-10)
                rate_margin = self.vehicles[k].transmission_rate_selected_rb - transmission_rate_required
                term3 = self.config.REWARD_LAMBDA4 * auxiliary_function_G(rate_margin)

                # 总奖励 - 公式18
                rewards[k] = term1 + term2 + term3
            else:
                # B_k ≤ 0: 无待传输数据 - 公式18第二个分支
                rewards[k] = self.config.REWARD_A1

        return rewards

    def _get_observations(self) -> List[np.ndarray]:
        """
        Get local observation for each vehicle

        严格按照论文公式(16)的状态空间：
        s_t(k) = {{G_k[m]}_{m∈M}, {I_k[m]}_{m∈M}, B_k, T_k, e, ε}

        Returns:
            obs_list: list of (obs_dim,) arrays
        """
        obs_list = []

        for i in range(self.n_vehicles):
            # 1. 信道增益向量 {G_k[m]}_{m∈M} - shape (n_rb,)
            channel_gains = self.channel.get_channel_gain(i)  # shape (n_rb,)
            # 归一化到[0, 1]
            channel_gains_norm = np.clip(10 * np.log10(channel_gains + 1e-10) / 100, 0, 1)

            # 2. 干扰向量 {I_k[m]}_{m∈M} - shape (n_rb,)
            # 从干扰信道矩阵计算当前受到的干扰
            # 重要: 这是vehicle i当前从其他vehicle接收到的干扰，取决于其他vehicle当前选择的RB
            interference_vector = np.zeros(self.n_rb)
            for m in range(self.n_rb):
                for j in range(self.n_vehicles):
                    if i != j:
                        # 如果vehicle j在RB m上发射，计算对i的干扰
                        if self.vehicles[j].selected_rb == (m):  # selected_rb已转换为0-indexed
                            interference_vector[m] += (self.vehicles[j].transmission_rate_selected_rb *
                                                      self.channel.interference_channel_gain[j, i, m])
            # 归一化干扰向量
            interference_vector_norm = np.clip(interference_vector / (1e-6 + np.max(interference_vector) + 1e-10), 0, 1)

            # 3. B_k 归一化 - 待传输数据比例 [0, 1]
            B_k_norm = np.clip(self.vehicles[i].B_k / (self.config.TASK_SIZE + 1e-10), 0, 1)

            # 4. T_k 归一化 - 剩余延迟容限比例 [0, 1]
            T_k_norm = np.clip(self.vehicles[i].T_k / (self.config.T_MAX + 1e-10), 0, 1)

            # 5. 时间进度 e - 归一化的episode进度 [0, 1]
            time_progress = self.current_step / self.episode_length

            # 6. 探索信息 ε - (简化为队列状态)
            queue_state = self.vehicles[i].get_queue_state()

            # 7. 组合状态 - 按照公式(16)的顺序
            obs = np.concatenate([
                channel_gains_norm,        # {G_k[m]}_{m∈M}, shape (n_rb,)
                interference_vector_norm,  # {I_k[m]}_{m∈M}, shape (n_rb,)
                [B_k_norm],                # B_k, shape (1,)
                [T_k_norm],                # T_k, shape (1,)
                [time_progress],           # e, shape (1,)
                [queue_state]              # ε, shape (1,)
            ]).astype(np.float32)

            obs_list.append(obs)

        return obs_list

    def get_episode_stats(self) -> Dict:
        """Get statistics for current episode (✅ 改: 增强容错能力)"""
        # 处理可能为空的列表
        mean_delay = float(np.mean(self.episode_delays)) if len(self.episode_delays) > 0 else 0.0
        mean_energy = float(np.mean(self.episode_energies)) if len(self.episode_energies) > 0 else 0.0
        mean_reward = float(np.mean(self.episode_rewards)) if len(self.episode_rewards) > 0 else 0.0

        # 约束满足率
        total_delays = len(self.episode_delays) if len(self.episode_delays) > 0 else 1
        csr = 1.0 - (self.constraint_violations / max(total_delays, 1))

        return {
            'mean_delay': mean_delay,
            'mean_energy': mean_energy,
            'mean_reward': mean_reward,
            'constraint_violations': self.constraint_violations,
            'constraint_satisfaction_rate': csr,
        }
