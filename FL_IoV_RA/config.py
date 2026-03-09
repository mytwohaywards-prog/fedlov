"""
Global configuration for HMDQN-based IoV Resource Allocation
Reference: IEEE paper on Distributed Resource Allocation With Federated Learning
for Delay-Sensitive IoV Services - HMDQN Framework
"""

# ============== Network Parameters ==============
# 严格按照论文参数（V2V通信）
N_VEHICLES = 10          # Number of vehicles (agents/VTLs)
N_RB = 5                 # Number of resource blocks (sub-channels)
BW_PER_RB = 5e6          # Hz, bandwidth per resource block (5 MHz per RB, total 25 MHz for 5 RBs)
MAX_POWER = 23           # dBm, maximum transmit power
NOISE_POWER_dBm = -114   # dBm, AWGN noise power
CARRIER_FREQUENCY = 2e9  # Hz, carrier frequency (2 GHz)

# ============== Computation Parameters ==============
# 严格按照论文Table 1参数（V2V负载 = 2×1060 Bytes）
MAX_CPU = 1e8            # Hz, maximum CPU frequency (100 MHz)
TASK_SIZE = 16960        # bits, task size (2×1060 bytes = 8480 bytes = 67840 bits, 用16960为中值)
CPU_CYCLES = 1e7         # cycles/task, computational complexity (10M cycles)
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
SHADOWING_STD = 8        # dB, log-normal shadowing std
FADING_MODEL = "rayleigh"  # "rayleigh" or "nakagami"
NAKAGAMI_M = 1           # Nakagami shape parameter (m=1 -> rayleigh)
PATH_LOSS_EXPONENT = 4   # path loss exponent (urban)
MIN_DISTANCE = 1         # meter, minimum distance to avoid singularity

# ============== DQN-specific Parameters (HMDQN) ==============
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
