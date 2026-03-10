# 深度分析：论文配置 vs 实现代码对标

## 第一部分：论文核心指标定义

### 论文报告的三个主要性能指标（Figure 3, Figure 4）

#### 1. Average Reward (Figure 3a, 3b)
- **范围**: 30 → 95（初期低，逐渐上升）
- **单位**: ？（论文未明确说明）
- **定义**: 每个episode的平均奖励

**关键问题**: 这是什么奖励函数？

#### 2. Network SEE (Figure 4a)
- **范围**: 1 → 8 (bps/Hz/W)
- **定义**: SEE = (C_V2I + C_V2V) / (BW * P_total)
- **公式**: SEE = Σ W*log2(1+SINR_k) / (N_RB * BW_per_RB * Σ P_k)

#### 3. V2V Success Rate (Figure 4c)
- **范围**: 30% → 95%
- **定义**: 满足SINR > 1dBm的V2V任务占比
- **公式**: Success = N_success / N_total

---

## 第二部分：论文的奖励函数逆向工程

从Figure 3和Figure 4的对应关系分析：

### 假设1: Reward = Success Rate
```
初期: Success_Rate ~ 30% => Reward ~ 30 ✓
最终: Success_Rate ~ 95% => Reward ~ 95 ✓
```
**这个假设成立！**

**推论**:
- 奖励函数 = V2V任务的成功率（百分比）
- 成功 = SINR >= 1dBm AND delay <= T_MAX
- 不是SEE或成本函数！

---

## 第三部分：论文的系统参数完整表（Table I + 实验设置）

### 3.1 网络参数

| 参数 | 论文值 | 单位 | 备注 |
|------|-------|------|------|
| 频率 fc | 2.0 ~ 4.7 | GHz | V2V典型范围，取2GHz |
| 总带宽 W | 3.6 | MHz | V2V共享带宽 |
| RB数量 N_RB | 5 | 个 | 假设每RB=0.72MHz |
| 最大功率 P_max | 23 | dBm | ≈ 200mW |
| 噪声功率 σ² | -114 | dBm | AWGN |
| SINR阈值 | 1 | dBm | V2V通信成功条件 |
| 延迟约束 T_max | 2.0 | s | 硬实时 |
| 车辆数 | 10 | 个 | 形成竞争资源 |

### 3.2 传输参数

| 参数 | 论文值 | 单位 | 备注 |
|------|-------|------|------|
| V2V负载 | 1-2 × 1060 | bytes | 8-16 Kbits |
| 计算复杂度 CPU_cycles | ? | cycles | 论文未明确 |
| 最大CPU频率 | 1e8 | Hz | 100 MHz |

### 3.3 DQN参数（Table II）

| 参数 | 论文值 | 备注 |
|------|-------|------|
| γ (discount) | 0.99 | 标准值 |
| α (learning rate) | 0.001 | RMSProp参数 |
| ε_start | 1.0 | 初始探索 |
| ε_end | 0.05 | 最终探索 |
| ε_decay_episodes | 200 | 800个episode的前200? 还是前800? |
| target_update_freq | 50 | 每50个episode |
| batch_size | 64 | 标准值 |
| buffer_size | ? | 未明确 |

### 3.4 奖励权重（关键！）

从论文文本：
- λ₁ = 0.1 (延迟权重)
- λ₂ = 0.9 (能源权重)
- λ₃ = 0.01 (资源利用权重)

**但这些是目标函数权重，不是奖励函数权重！**

---

## 第四部分：现有实现的问题诊断

### 4.1 当前奖励设计
```python
# 我们目前用的: 基于成本的奖励
cost = λ₁ * delay + λ₂ * energy + constraint_penalty
reward = 100 - cost
```

**问题**:
- 这给出初期reward ~ 74-78 (太高！)
- 论文给出初期reward ~ 30

### 4.2 差异分析
```
论文初期reward = 30 => 初期成功率 = 30%
我们初期reward = 75 => ???

差异原因可能:
1. 初期SINR太低 => 初期成功率低 => 奖励低
2. 我们的SINR初期太高 => 初期成功率高 => 奖励高
```

### 4.3 SINR计算差异
```
论文: SINR = P_rx / (I_interference + σ²_noise)
      需要精确计算干扰功率

我们: 简化的SINR计算，可能高估了信号强度或低估了干扰

结果: 我们的初期SINR偏高 => 成功率偏高 => 奖励偏高
```

---

## 第五部分：正确的奖励函数

### 修正方案

```python
def compute_reward_correct(sinrs, delays):
    """
    基于V2V Success Rate的奖励函数

    success = (SINR >= SINR_threshold dB) AND (delay <= T_MAX)
    reward = 100 * (success_count / total_count)

    范围: [0, 100]
    初期: ~30 (30%车辆成功)
    最终: ~95 (95%车辆成功)
    """
    SINR_threshold_dB = 1  # dBm
    SINR_threshold_linear = 10 ** (SINR_threshold_dB / 10)  # ~1.26
    T_MAX = 2.0  # s

    # 计算成功数
    sinr_satisfied = sinrs >= SINR_threshold_linear
    delay_satisfied = delays <= T_MAX
    success = sinr_satisfied & delay_satisfied  # 必须同时满足

    # 奖励 = 成功率 × 100
    reward = 100 * np.sum(success) / len(success)

    return reward
```

### 关键点
1. **奖励直接映射到成功率** (不是SEE，不是成本函数)
2. **成功的定义**: SINR AND delay 同时满足
3. **范围**: 0-100，自然对应30-95%

---

## 第六部分：实现修正清单

### 必改项

1. **奖励函数** ✗
   - 当前: 基于成本 => 初期高
   - 应改: 基于Success Rate => 初期低

2. **SINR计算** ✓ (暂时接受)
   - 需要验证初期SINR是否过高
   - 检查干扰计算

3. **延迟计算** ✓ (应该正确)

4. **能耗计算** ✓ (应该正确)

### 可选项

1. 论文Figure 3对应哪个算法？ (FMDQN, distDQN, 还是HMDQN?)
2. 是否需要不同的baseline对比?

---

## 第七部分：预期修正结果

修改后预期：
```
初期 (Episode 1-50):
- Reward: 25-35 (初期只有30%车成功)
- Success Rate: 30-40%
- SINR满足: ~30%

中期 (Episode 50-150):
- Reward: 50-75 (50-75%车成功)
- Success Rate: 50-75%
- SINR满足: ~50-75%

后期 (Episode 150-200):
- Reward: 80-95 (80-95%车成功)
- Success Rate: 80-95%
- SINR满足: ~80-95%
```

**这将匹配论文Figure 3(a)！**
