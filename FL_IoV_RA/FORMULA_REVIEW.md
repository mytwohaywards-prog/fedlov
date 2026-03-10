# 严格对标论文公式的代码审查 (Code Review Against Paper Formulas)

## 审查范围
代码组件与论文公式对标逐一检查。发现的问题严格按照IEEE论文要求进行修复。

---

## 1. SINR计算 (Formulas 1, 5, 6)

### 论文要求

**Formula (5) - V2V SINR:**
$$\gamma_k^v[m] = \frac{P_k^v[m]g_k[m]}{I_k[m] + \sigma^2}$$

其中：
- $P_k^v[m]$: Vehicle k在RB m上的发射功率
- $g_k[m]$: Vehicle k到接收端的信道增益 (desired channel)
- $I_k[m]$: 总干扰功率，来自其他车辆在同一RB上的发射
- $\sigma^2$: 噪声功率

**Formula (6) - 干扰计算:**
$$I_k[m] = P_m^c[m]\bar{g}_{m,k}[m] + \sum_{k' \in K \setminus k} \rho_{k'}[m]P_{k'}^v[m]\bar{g}_{k',k}[m]$$

其中：
- $\bar{g}_{m,k}[m]$: V2I到V2V的干扰信道
- $\bar{g}_{k',k}[m]$: 从Vehicle k'到Vehicle k的干扰信道

### 代码实现 (channel.py:196-236)

```python
def compute_sinr(self, power_linear: np.ndarray, rb_indices: np.ndarray):
    # ...
    for i in range(self.n_vehicles):
        rb_idx = int(rb_indices[i])

        # 期望信号
        if rb_idx > 0 and rb_idx <= self.n_rb:
            desired_power = power_linear[i] * self.channel_gain[i, rb_idx - 1]
        else:
            desired_power = power_linear[i] * np.mean(self.channel_gain[i])

        # 干扰计算
        interference_power = 0
        for j in range(self.n_vehicles):
            if i != j:
                rb_idx_j = int(rb_indices[j])
                if rb_idx == rb_idx_j and rb_idx > 0:
                    interference_power += power_linear[j] * self.channel_gain[j, rb_idx - 1]  # ❌
```

### 审查结论

❌ **严重错误: 干扰信道增益计算不正确**

**问题1:**
- **代码行:** Line 229 使用 `self.channel_gain[j, rb_idx - 1]`
- **应该:** `self.channel_gain[i, rb_idx - 1]` 或专门的干扰信道矩阵
- **原因:** 干扰是从Vehicle j的发射端到Vehicle i的接收端的信道增益
  - $\bar{g}_{k',k}[m]$ = 从k'到k的信道增益
  - 当前代码使用 $g_j[m]$ (从j到j的增益) 这是错误的
  - 应该是从j到i的增益，即 $\bar{g}_{j,i}[m]$

**问题2:**
- 干扰计算缺少V2I项: V2I基站的干扰 $P_m^c[m]\bar{g}_{m,k}[m]$
- 在纯V2V场景中这可能被忽略，但应该有注释说明

### 修复建议

需要创建干扰信道矩阵 $\bar{G}_{ij}$ 表示从Vehicle i到Vehicle j的干扰信道增益。

---

## 2. 传输速率 (Formulas 4, 7)

### 论文要求

**Formula (7) - V2V传输速率:**
$$C_k^v[m] = W \log_2(1 + \gamma_k^v[m])$$

其中 W = 每RB的带宽 (Hz)

### 代码实现 (channel.py:238-251)

```python
def compute_rate(self, sinr: np.ndarray, bandwidth: np.ndarray):
    rate = bandwidth * np.log2(sinr + 1)
    return rate
```

### 审查结论

✅ **正确** - 严格按照Shannon容量公式实现

---

## 3. RB分配约束 (Formula 2, 3)

### 论文要求

**Formula (3) - RB选择约束:**
$$\sum_{m \in M} \rho_k[m] = 1, \forall k \in K$$

**含义:** 每个Vehicle k必须选择恰好一个RB进行传输（one-to-one约束）

### 代码实现 (iov_env.py:263-310)

```python
def _decode_discrete_actions(self, actions):
    for i in range(self.n_vehicles):
        action = actions[i]
        rb_idx = action // (self.n_power_actions * self.n_cpu_actions)

        # RB分配
        if rb_idx > 0 and rb_idx <= self.n_rb:
            bandwidth[i] = self.config.BW_PER_RB
        else:
            bandwidth[i] = self.config.BW_PER_RB * 0.1  # 最小分配
```

### 审查结论

❌ **严重错误: 违反RB约束**

**问题:**
- rb_idx = 0 导致 `bandwidth[i] = 0.1 × BW_PER_RB` (最小带宽)
- 这允许Vehicle不选择任何RB（或选择无效RB）
- 违反 Formula (3) 的约束: $\sum_{m \in M} \rho_k[m] = 1$
- 每个Vehicle **必须** 选择一个有效的RB (1到N_RB)

### 修复建议

1. 将RB_ACTIONS改为N_RB (不包含"无RB"选项)
2. 每个vehicle的rb_idx范围应该是 0 到 N_RB-1 (对应RB 1到N_RB)
3. 删除条件，直接映射：`bandwidth[i] = BW_PER_RB` 对所有有效action

---

## 4. 状态空间 (Formula 15, 16)

### 论文要求

**Formula (16) - 完整状态空间:**
$$s_t(k) = \{\{G_k[m]\}_{m \in M}, \{I_k[m]\}_{m \in M}, B_k, T_k, e, \epsilon\}$$

其中：
- $\{G_k[m]\}_{m \in M}$: 所有RB的信道增益向量 (维度 N_RB)
- $\{I_k[m]\}_{m \in M}$: 所有RB的干扰向量 (维度 N_RB)
- $B_k$: 待传输数据 (bits)
- $T_k$: 剩余延迟容限 (seconds)
- $e$: 当前时隙编号 或 归一化的episode进度
- $\epsilon$: 探索相关信息

### 代码实现 (iov_env.py:404-441)

```python
def _get_observations(self):
    obs_list = []
    for i in range(self.n_vehicles):
        channel_gains = self.channel.get_channel_gain(i)  # shape (n_rb,) ✓
        channel_gains_norm = np.clip(10 * np.log10(channel_gains + 1e-10) / 100, 0, 1)

        queue_state = self.vehicles[i].get_queue_state()  # 不对应 B_k
        interference_est = np.mean(self.interference_coord.interference_matrix[i])  # ❌ 标量，应该是向量
        deadline_info = 0.5  # placeholder ❌

        obs = np.concatenate([
            channel_gains_norm,        # {G_k[m]} ✓
            [queue_state],             # ❌ 不是 B_k
            [deadline_info],           # ❌ 不是 T_k
            [interference_est_norm]    # ❌ 应该是 {I_k[m]} 向量
        ])
```

### 审查结论

❌ **严重错误: 状态空间不完整且定义错误**

**问题1: 缺少核心变量 B_k 和 T_k**
- 当前: `queue_state` (队列归一化)
- 应该: `B_k` (待传输比特数) 和 `T_k` (剩余延迟容限)
- 影响: Agent无法准确学习传输速率约束

**问题2: 干扰表示不正确**
- 当前: 标量 `np.mean(interference_matrix[i])`
- 应该: 向量 `interference_matrix[i]` (维度 N_RB)
- 影响: 失去每个RB的干扰信息

**问题3: 缺少时间信息 e 和 ε**
- 当前: 不包含
- 应该: 包含episode进度或时隙索引
- 原因: Agent需要了解当前时间位置

### 修复建议

```python
def _get_observations(self):
    for i in range(self.n_vehicles):
        channel_gains_norm = ...          # {G_k[m]}_{m∈M}
        interference_vector_norm = ...   # {I_k[m]}_{m∈M} - 完整向量
        B_k_norm = self.vehicles[i].B_k / self.config.TASK_SIZE  # B_k归一化
        T_k_norm = self.vehicles[i].T_k / self.config.T_MAX       # T_k归一化
        time_progress = self.current_step / self.episode_length   # 时间进度 e

        obs = np.concatenate([
            channel_gains_norm,        # 维度: N_RB
            interference_vector_norm,  # 维度: N_RB
            [B_k_norm],
            [T_k_norm],
            [time_progress]
        ])
```

---

## 5. 奖励函数 (Formulas 18, 19)

### 论文要求

**Formula (18) - 完整奖励:**
$$r_t(k) = \begin{cases}
\zeta^{net} + \lambda_3 G(\gamma_k^c - \gamma^{th}) + \lambda_4 G(\sum_{m \in M} \rho_k[m]C_k^v[m] - \frac{B_k}{T_k}), & \text{if } B_k > 0 \\
A_1, & \text{otherwise}
\end{cases}$$

**Formula (19) - 从属函数:**
$$G(x) = \begin{cases}
A_2, & \text{if } x > 0 \\
x, & \text{otherwise}
\end{cases}$$

### 代码实现 (iov_env.py:312-402)

✅ **刚完成的实现 - 严格遵循公式**

已正确实现：
- B_k > 0 条件检查 ✓
- 三项奖励求和 ✓
- G(x)从属函数 ✓
- A_1惩罚 ✓

---

## 6. 传输速率约束 (Formula 12)

### 论文要求

**Formula (12) - 传输速率约束:**
$$\sum_{m \in M} \rho_k[m]C_k^v[m] \geq \frac{B_k}{T_k}, \forall k \in K$$

**由于Formula (3) 的约束 $\sum_{m \in M} \rho_k[m] = 1$:**
- 每个k只选一个RB，所以这个约束简化为：
$$C_k^v[m_{selected}] \geq \frac{B_k}{T_k}$$

### 代码实现 (iov_env.py:393-395)

```python
transmission_rate_required = self.vehicles[k].B_k / (self.vehicles[k].T_k + 1e-10)
rate_margin = self.vehicles[k].transmission_rate_selected_rb - transmission_rate_required
term3 = self.config.REWARD_LAMBDA4 * auxiliary_function_G(rate_margin)
```

### 审查结论

✅ **正确** - 根据简化约束正确实现

---

## 7. B_k 和 T_k 更新

### 论文要求

**Formula (12)中的含义:**
- $B_k$: 待传输的剩余数据 (初值 = TASK_SIZE)
- $T_k$: 剩余延迟容限 (初值 = T_MAX)
- 每个时隙：$B_k$ 减少已发送比特，$T_k$ 减少经过的时间

### 代码实现 (iov_env.py:229-243)

```python
# Update B_k
bits_completed = transmission_rate * self.config.SLOT_DURATION
for i in range(self.n_vehicles):
    self.vehicles[i].B_k = max(0, self.vehicles[i].B_k - bits_completed[i])

# Update T_k
for i in range(self.n_vehicles):
    self.vehicles[i].T_k = max(0, self.vehicles[i].T_k - self.config.SLOT_DURATION)
```

### 审查结论

✅ **正确** - 按照公式正确递减

---

## 8. SEE计算 (Formulas 8, 10, 11)

### 论文要求

**Formula (8) - V2I SEE:**
$$\zeta^{V2I} = \frac{\sum_{m \in M} C_m^c[m]}{\sum_{m \in M} W \cdot (M \cdot P_c + \sum_{m \in M} P_m^c[m])}$$

**Formula (10) - V2V SEE:**
$$\zeta^{V2V} = \frac{\sum_{m \in M} \sum_{k \in K} C_k^v[m]}{\sum_{m \in M} W \cdot (\sum_{m \in M} K \cdot P_c + \sum_{k} P_k^v[m])}$$

**Formula (11) - 网络SEE:**
$$\zeta^{net} = \lambda_1 \zeta^{V2I} + \lambda_2 \zeta^{V2V}$$

### 代码实现 (iov_env.py:347-357)

```python
total_v2v_capacity = np.sum(rates)
total_energy = np.sum(energies) + 1e-10
BW_total = self.config.N_RB * self.config.BW_PER_RB

see_v2v = total_v2v_capacity / (BW_total * total_energy)
see_v2i = see_v2v  # 简化

zeta_net = (self.config.REWARD_LAMBDA1_SEE * see_v2i +
            self.config.REWARD_LAMBDA2_SEE * see_v2v)
```

### 审查结论

⚠️ **部分正确但简化过度**

**问题:**
- SEE的分母应该包含电路功耗 $P_c$ 项（多项）
- 当前实现过于简化
- V2I SEE应单独计算而非等于V2V SEE
- 在纯V2V场景下，可以简化，但应有清晰注释

---

## 总结 (Summary of Critical Issues)

| # | 模块 | 公式 | 问题 | 严重程度 |
|----|------|------|------|---------|
| 1 | SINR干扰 | (5), (6) | 干扰信道增益计算错误 | 🔴 严重 |
| 2 | RB约束 | (3) | 允许无效RB选择 | 🔴 严重 |
| 3 | 状态空间 | (16) | 缺少B_k, T_k，干扰为标量 | 🔴 严重 |
| 4 | SEE计算 | (8,10,11) | 过度简化，缺少电路功耗项 | 🟡 中等 |
| 5 | 奖励函数 | (18), (19) | ✅ 已修复 | - |
| 6 | 传输速率约束 | (12) | ✅ 正确 | - |
| 7 | B_k/T_k更新 | - | ✅ 正确 | - |

---

## 立即修复计划

### Phase 1: 关键修复 (必须立即修复)
1. **修复干扰信道增益** (channel.py)
2. **修复RB约束** (iov_env.py action decoding)
3. **完善状态空间** (iov_env.py observations)

### Phase 2: 参数调整
4. **改进SEE计算** (add circuit power)

### Phase 3: 验证
5. 运行单元测试验证所有公式实现
