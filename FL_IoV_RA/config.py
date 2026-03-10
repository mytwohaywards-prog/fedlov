"""
Global configuration for HMDQN-based IoV Resource Allocation
Reference: IEEE paper on Distributed Resource Allocation With Federated Learning
for Delay-Sensitive IoV Services - HMDQN Framework
"""

# ============== Network Parameters ==============
# 严格按照论文参数（V2V通信 - Table I & II）
N_VEHICLES = 10          # Number of vehicles (agents/VTLs)
N_RB = 5                 # Number of resource blocks (sub-channels)

# 论文Table I: W = 3.6 MHz (总V2V带宽)
# 5个RB分配: 3.6MHz / 5 = 0.72MHz per RB
BW_PER_RB = 0.72e6       # Hz, bandwidth per resource block (0.72 MHz, total 3.6 MHz) ✓ 修正

# 论文Table I: 频率为4.7 GHz (V2V通信频带)
CARRIER_FREQUENCY = 4.7e9  # Hz, carrier frequency (4.7 GHz, NOT 2GHz) ✓ 修正

MAX_POWER = 23           # dBm, maximum transmit power (论文Table I)
NOISE_POWER_dBm = -114   # dBm, AWGN noise power (论文Table I)

# ============== Computation Parameters ==============
# 严格按照论文Table I参数（V2V负载）
MAX_CPU = 1e8            # Hz, maximum CPU frequency (100 MHz, 论文Table I)
# V2V payload = 2 × 1060 bytes = 2120 bytes = 16960 bits (最大负载)
TASK_SIZE = 16960        # bits, task size for V2V communication
CPU_CYCLES = 1e8         # cycles/task, computational complexity per task
TASK_ARRIVAL_RATE = 0.5  # tasks per time slot, poisson rate

# ============== Delay and Cost Constraints (严格遵循论文Table 3) ==============
T_MAX = 2.0              # seconds, maximum allowed latency
LAMBDA1 = 0.1            # ✅ 改: 1.0→0.1, 延迟权重（小）, 论文要求
LAMBDA2 = 0.9            # ✅ 改: 0.1→0.9, 能源权重（大）, 论文要���：能效优先策略
LAMBDA3 = 0.01           # weight for resource utilization

# ============== Vehicular Mobility ==============
SCENARIO_LENGTH = 1000   # meters, road length
SCENARIO_WIDTH = 50      # meters, road width
V_MAX = 30               # m/s, maximum vehicle speed
SLOT_DURATION = 0.1      # seconds, time slot duration

# ============== Channel Model ==============
# 论文Table II指定不同条件下的阴影标准差
SHADOWING_STD = 3        # dB, log-normal shadowing std for LOS (论文Table II: V2V LOS=3dB) ✓ 修正
SHADOWING_STD_NLOS = 4   # dB, shadowing std for NLOS (论文Table II: V2V NLOS=4dB)

# ========== 奖励函数参数（论文公式18-19） ==========
# 严格按照论文Section III.A "Key Parameters Design of FMDQN"
# 奖励函数: r_t(k) = ζ^net + λ₃G(γ_k^c - γ^th) + λ₄G(∑ρ_k[m]C_k^c[m] - B_k/T_k)
# 从属函数: G(x) = A₂ if x>0 else x

REWARD_LAMBDA3 = 0.5     # λ₃: SINR约束的权重 (论文未明确，需通过实验调整)
REWARD_LAMBDA4 = 0.5     # λ₄: 传输速率约束的权重 (论文未明确，需通过实验调整)
REWARD_A1 = -5.0         # A₁: 无数据时的固定奖励 (负值惩罚)
REWARD_A2 = 1.0          # A₂: 约束满足时的固定奖励增益

# SEE权重 (论文公式11: ζ^net = λ₁ζ^V2I + λ₂ζ^V2V)
REWARD_LAMBDA1_SEE = 0.5 # λ₁: V2I SEE的权重
REWARD_LAMBDA2_SEE = 0.5 # λ₂: V2V SEE的权重

# ========== SINR阈值参数 ==========
# 论文Table I: SINR_threshold = 1 dBm (严格模式)
# 测试模式: SINR_threshold = 0.0 dBm (放宽，用于快速验证)
SINR_THRESHOLD_dB = 1.0  # ← 恢复为论文要求的1.0 dBm
FADING_MODEL = "rayleigh"  # "rayleigh" or "nakagami" (论文Table II: Rice/Rayleigh)
NAKAGAMI_M = 1           # Nakagami shape parameter (m=1 -> rayleigh)
PATH_LOSS_EXPONENT = 3.8 # path loss exponent (根据3GPP模型确定)
MIN_DISTANCE = 1         # meter, minimum distance to avoid singularity
DECORRELATION_DISTANCE = 10  # meters, decorrelation distance (论文Table II)

# ============== FMDQN-specific Parameters ==============
# Federated Learning Configuration (Algorithm 1)
AGGREGATION_FREQ = 50                # 每50个episode执行一次联邦聚合 (Algorithm 1 Line 14)
AGGREGATION_WEIGHTS = "equal"        # 聚合权重: "equal" = 所有agents等权重 (1/K)

# Discrete Action Spaces
N_RB_ACTIONS = N_RB + 1              # RB selection: 0 to N_RB (0=no RB)
N_POWER_ACTIONS = 5                  # Power levels: 0 to 4 (0=0%, 4=100%)
N_CPU_ACTIONS = 5                    # CPU frequency levels: 0 to 4
N_ACTIONS_TOTAL = N_RB_ACTIONS * N_POWER_ACTIONS * N_CPU_ACTIONS  # Total discrete actions

# Reinforcement Learning (DQN) - 严格遵循论文Table 3参数
GAMMA = 0.99             # discount factor
BATCH_SIZE = 64          # replay buffer batch size
BUFFER_SIZE = 30000      # max replay buffer capacity (✅ 改: 100000→30000, 论文要求)
LEARNING_RATE = 1e-3     # RMSProp learning rate for Q-network (0.001, 论文要求)
TAU = 0.01               # soft update coefficient for target network
EPSILON_START = 1.0      # initial exploration rate
EPSILON_END = 0.05       # final exploration rate (ε_min = 0.02, 近似设为0.05)
EPSILON_DECAY = 800      # epsilon decay episodes (✅ 改: 200→800, 论文要求前800个回合线性衰减)
UPDATE_FREQ = 4          # update Q-network every N steps
TARGET_UPDATE_FREQ = 200 # update target network every N updates (✅ 改: 硬更新频率, 每8个Episode=200÷4≈50步)

# ============== Federated Learning ==============
FL_ROUNDS = 200          # total federated learning rounds
LOCAL_STEPS = 10         # local training steps per FL round
FL_PARTICIPATION_RATE = 1.0  # participation rate (1.0 = all agents)
FL_AGGREGATION = "FedAvg"    # aggregation method

# ============== Training Configuration ==============
NUM_EPISODES = FL_ROUNDS
RANDOM_SEED = 42
DEVICE = "cpu"  # "cpu" or "cuda"
USE_NORMALIZATION = True
EXPLORATION_START = 1.0
EXPLORATION_END = 0.05
EXPLORATION_STEPS = FL_ROUNDS * LOCAL_STEPS * 0.3  # decay over 30% of training

# ============== Evaluation Configuration ==============
EVAL_INTERVAL = 10       # evaluate every N FL rounds
EVAL_EPISODES = 5        # episodes per evaluation
RENDER = False
SAVE_INTERVAL = 50       # save models every N FL rounds

# ============== Baseline Configurations ==============
BASELINE_EQUAL_BW = 1.0 / N_RB  # equal bandwidth ratio per RB
BASELINE_EQUAL_POWER = 1.0 / N_VEHICLES  # equal power ratio
BASELINE_EQUAL_CPU = 1.0 / N_VEHICLES  # equal CPU ratio

# ============== Logging ==============
LOG_INTERVAL = 1         # log every N FL rounds
VERBOSE = True
SAVE_RESULTS = True
RESULTS_DIR = "results/"
MODELS_DIR = "models/"

# ============== Helper Function ==============
def get_config_dict():
    """Return all config as dictionary"""
    import sys
    current_module = sys.modules[__name__]
    config_dict = {
        key: getattr(current_module, key)
        for key in dir(current_module)
        if not key.startswith('_') and key.isupper()
    }
    return config_dict
