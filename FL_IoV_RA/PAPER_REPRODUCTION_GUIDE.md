# 论文精确复现指南

## 1. 核心指标定义（来自论文公式8-12）

### 1.1 Spectral Energy Efficiency (SEE)
```
SEE = Σ(C_V2I^k) + Σ(C_V2V^k) / Σ(P_total^k)  [bps/Hz/W]

其中：
- C_V2I^k = W * log2(1 + SINR_V2I^k)  [V2I容量]
- C_V2V^k = W * log2(1 + SINR_V2V^k)  [V2V容量]
- P_total^k = P_circuit + P_tx^k  [总功耗]
```

### 1.2 V2V Success Rate
```
SuccessRate = N_success / N_total

其中：
- N_success = 满足 SINR_k >= SINR_threshold (1dBm) 的任务数
- N_total = 总V2V任务数
```

### 1.3 综合奖励函数
```
reward = α * SEE/SEE_max + β * SuccessRate  [0-100 scale]

推荐权重：
- α = 0.5  (SEE权重)
- β = 0.5  (Success Rate权重)
- 归一化：SEE_max ≈ 10 bps/Hz/W
```

## 2. 系统参数（Table I + V2V场景）

### 2.1 网络参数（论文Table I）
```
频率 fc = 4.7 GHz (V2V通常2-6GHz，可简化为2GHz)
带宽 W = 3.6 MHz (5G NR标准)
RB数量 N_RB = 5 (5×3.6MHz = 18MHz总带宽)
V2V传输功率 P_tx ∈ {23, 15, 10, 5, 0} dBm  (5个电平)
最大功率 P_max = 23 dBm
噪声功率 σ² = -114 dBm
SINR阈值 = 1 dBm (V2V任务成功判定)
```

### 2.2 计算参数
```
车辆数 N_vehicles = 10
V2V负载 = [1, 2] × 1060 bytes (对应16960 bits中值)
计算复杂度 CPU_cycles = 1e7 ~ 1e8 cycles
最大CPU频率 = 1e8 Hz (100 MHz)
```

### 2.3 时间参数
```
时隙时长 SLOT_DURATION = 0.1 s
最大延迟 T_MAX = 2.0 s
信道更新周期 = 100 ms (每个时隙更新一次)
```

### 2.4 衰减模型参数（论文Table I补充）
```
路径损耗模型：  PL(dB) = 38.77 + 16.7*log10(d) + 18.2*log10(fc)
阴影标准差：    σ_shadow = 4 dB (V2V on same road)
衰落模型：      Rice/Rayleigh fading (K因子基于LoS/NLoS)
更新率：        每1ms更新一次快速衰落
```

## 3. DQN参数配置（论文明确指定）

```
γ = 0.99  (discount factor)
α = 0.001  (learning rate, RMSProp)
ε_start = 1.0
ε_end = 0.05
ε_decay_episodes = 200  (前200个episode线性衰减)
target_update_freq = 50  (每50个episode更新一次)
batch_size = 64
buffer_size = 30000
optimizer = RMSProp (α=0.99, ε=1e-8)
```

## 4. 奖励函数实现代码

```python
def compute_reward_paper_compliant(self, delays, energies, rates, sinrs):
    """
    严格遵循论文公式8-12的奖励函数

    Args:
        delays: [n_vehicles] 传输延迟
        energies: [n_vehicles] 能耗
        rates: [n_vehicles] 传输速率
        sinrs: [n_vehicles] SINR值

    Returns:
        rewards: [n_vehicles] 奖励值 range [0, 100]
    """
    # 1. 计算SEE (Spectral Energy Efficiency)
    total_capacity = np.sum(rates)  # bps
    total_power = np.sum(energies)  # W

    BW = 3.6e6  # Hz (5 RBs × 0.72MHz each, or 3.6MHz total)
    see = total_capacity / (BW * (total_power + 1e-8))  # bps/Hz/W
    see_normalized = min(100, see / 10.0 * 100)  # 归一化到0-100

    # 2. 计算V2V Success Rate
    sinr_threshold_dB = 1  # dBm
    sinr_threshold_linear = 10 ** (sinr_threshold_dB / 10)

    success_count = np.sum(sinrs > sinr_threshold_linear)
    total_tasks = len(sinrs)
    success_rate = success_count / total_tasks * 100  # 0-100

    # 3. 综合奖励
    alpha = 0.5  # SEE权重
    beta = 0.5   # Success Rate权重

    reward = alpha * see_normalized + beta * success_rate

    return reward
```

## 5. 验证检查清单

- [ ] SEE从低(<5)上升到高(>8)
- [ ] Success Rate从低(<30%)上升到高(>80%)
- [ ] Reward从低(<20)上升到高(>70) - 匹配Figure 3
- [ ] 延迟满足率(CSR) > 80%
- [ ] 能耗合理 (< 0.05J per slot)
- [ ] Epsilon正确衰减：1.0 → 0.05 over 200 episodes
- [ ] 无数值不稳定 (NaN, overflow, RuntimeWarning)

## 6. 参数对标（与论文Figure 3）

**训练早期 (Episode 1-50)**
- Reward: 10-30
- Success Rate: 20-40%
- SEE: 2-4 bps/Hz/W

**训练中期 (Episode 51-150)**
- Reward: 30-60
- Success Rate: 50-70%
- SEE: 4-7 bps/Hz/W

**训练后期 (Episode 151-200)**
- Reward: 60-85
- Success Rate: 80-95%
- SEE: 7-10 bps/Hz/W

## 7. 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Reward不改进 | 学习率过低或过高 | 调整LR: 1e-4 ~ 1e-2 |
| Reward波动剧烈 | Buffer过小或batch过大 | 增大buffer到50K, batch=32 |
| SEE计算NaN | 功耗为0或速率为负 | 加1e-8防护项 |
| Success Rate卡在0% | 初始SINR极低 | 增加初始功率或减少噪声 |
