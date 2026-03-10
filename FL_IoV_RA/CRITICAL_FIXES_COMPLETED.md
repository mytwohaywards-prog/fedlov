# 严格对标论文公式的代码修正完成 (Critical Fixes Implementation Complete)

## 修复概要 (Summary)

根据用户提供的论文截图和公式，我对代码进行了逐一审查，发现了3个严重错误和1个中等错误。已完成3个严重错误的修复：

| 项 | 错误 | 严重程度 | 状态 |
|-----|------|---------|------|
| 1 | 干扰信道增益计算错误 (Formulas 5, 6) | 🔴 严重 | ✅ 已修复 |
| 2 | RB分配约束违反 (Formula 3) | 🔴 严重 | ✅ 已修复 |
| 3 | 状态空间不完整 (Formula 16) | 🔴 严重 | ✅ 已修复 |
| 4 | SEE计算过度简化 (Formula 8, 10, 11) | 🟡 中等 | ⏳ 待优化 |

---

## 修复详情

### 修复1: 干扰信道增益计算 (Formulas 5, 6)

#### 问题
**File:** `env/channel.py:229` (修复前)
```python
interference_power += power_linear[j] * self.channel_gain[j, rb_idx - 1]  # ❌ 错误
```
- 使用 `channel_gain[j, j]` (vehicle j自己到j的增益)
- 应该是 `channel_gain[j->i]` (vehicle j到vehicle i的干扰增益)

#### 解决方案
1. **新增干扰信道矩阵** (channel.py):
   ```python
   self.interference_channel_gain = np.ones((n_vehicles, n_vehicles, n_rb))
   # interference_channel_gain[j, i, rb] = 从j到i的信道增益
   ```

2. **新增方法计算干扰信道** (channel.py):
   ```python
   def _compute_interference_channels(self):
       # 基于i-j之间的距离，为每对(j,i)计算干扰信道增益
       # 严格遵循3GPP路径损耗模型
   ```

3. **修正SINR计算** (channel.py:254-300):
   ```python
   def compute_sinr(...):
       # 公式(5): γ_k^v[m] = P_k^v[m]g_k[m] / (I_k[m] + σ²)
       # 公式(6): I_k[m] = Σ_{k'∈K\k} ρ_{k'}[m]P_{k'}^v[m]ḡ_{k',k}[m]

       for j in range(self.n_vehicles):
           if i != j:
               # 正确使用干扰信道矩阵
               interference_power += power_linear[j] * self.interference_channel_gain[j, i, rb_array_idx]
   ```

#### 影响
- ✅ SINR计算现在物理上正确
- ✅ 干扰功率准确反映了其他vehicle的发射对当前vehicle的影响
- ✅ 与论文公式(5)(6)严格对应

---

### 修复2: RB分配约束违反 (Formula 3)

#### 问题
**File:** `env/iov_env.py:294-300` (修复前)
```python
if rb_idx > 0 and rb_idx <= self.n_rb:
    bandwidth[i] = self.config.BW_PER_RB
else:
    bandwidth[i] = self.config.BW_PER_RB * 0.1  # 最小分配 ❌ 违反约束
```
- 允许 `rb_idx = 0` (无RB)
- 违反论文公式(3): `∑_m ρ_k[m] = 1` (每个vehicle必须选一个RB)

#### 解决方案
**File:** `env/iov_env.py:263-310` 修改后的 `_decode_discrete_actions()`:
```python
# 强制每个vehicle选择有效的RB (1到N_RB)
rb_idx = (rb_idx % self.n_rb) + 1  # 范围: 1到N_RB

# 所有vehicle都获得BW_PER_RB (遵循约束3)
bandwidth[i] = self.config.BW_PER_RB
```

#### 影响
- ✅ 每个vehicle总是选择恰好一个有效RB
- ✅ 严格满足约束(3): `∑ ρ_k[m] = 1`
- ✅ 简化了action空间的处理，减少了无效RB选择

---

### 修复3: 状态空间不完整 (Formula 16)

#### 问题
**论文要求 (Formula 16):**
$$s_t(k) = \{\{G_k[m]\}_{m \in M}, \{I_k[m]\}_{m \in M}, B_k, T_k, e, \epsilon\}$$

**修复前的观测:**
```python
obs = np.concatenate([
    channel_gains_norm,        # {G_k[m]} ✓
    [queue_state],             # ❌ 不是B_k
    [deadline_info],           # ❌ 不是T_k (硬编码0.5)
    [interference_est_norm]    # ❌ 标量，应该是向量 {I_k[m]}
])  # 维度: 8 (不准确)
```

#### 解决方案
**File:** `env/iov_env.py:404-466` 完全重写 `_get_observations()`:

```python
def _get_observations(self):
    for i in range(self.n_vehicles):
        # 1. {G_k[m]}_{m∈M} - shape (n_rb,)
        channel_gains_norm = ...          # ✓

        # 2. {I_k[m]}_{m∈M} - shape (n_rb,) - 完整干扰向量
        interference_vector = np.zeros(self.n_rb)  # ✓ 改为向量而非标量
        for m in range(self.n_rb):
            for j in range(self.n_vehicles):
                if self.vehicles[j].selected_rb == m:
                    interference_vector[m] += ...

        # 3. B_k 归一化 [0, 1]
        B_k_norm = self.vehicles[i].B_k / self.config.TASK_SIZE  # ✓

        # 4. T_k 归一化 [0, 1]
        T_k_norm = self.vehicles[i].T_k / self.config.T_MAX  # ✓

        # 5. 时间进度 e
        time_progress = self.current_step / self.episode_length  # ✓

        # 6. 队列状态 ε
        queue_state = self.vehicles[i].get_queue_state()  # ✓

        obs = np.concatenate([
            channel_gains_norm,        # n_rb
            interference_vector_norm,  # n_rb
            [B_k_norm],
            [T_k_norm],
            [time_progress],
            [queue_state]
        ])  # 维度: 2*n_rb + 4 = 2*5 + 4 = 14
```

#### 观测维度更新
- **修复前:** `obs_dim = N_RB + 3 = 8`
- **修复后:** `obs_dim = 2*N_RB + 4 = 14`

#### 影响
- ✅ 状态空间现在包含所有必要信息 (按公式16)
- ✅ Agent能准确学习关于B_k和T_k的传输速率约束
- ✅ 完整的干扰信息向量让Agent能针对每个RB的干扰做决策
- ✅ 时间信息让Agent学会时间意识

---

## 其他改进

### 文件: `env/iov_env.py`
- ✅ 已正确实现奖励函数 (Formula 18, 19) - 上一步完成
- ✅ B_k和T_k的递减逻辑正确 (公式要求)
- ✅ 传输速率约束检查正确 (Formula 12)

### 文件: `config.py`
- ✅ 所有参数按论文Table I, II正确设置
- ✅ SINR阈值 = 1.0 dBm (hard constraint)
- ✅ 带宽和频率参数正确

---

## 代码变更统计

| 文件 | 改动 | 行数 |
|------|------|------|
| `env/channel.py` | 新增干扰信道矩阵，重写compute_sinr | +60 |
| `env/iov_env.py` | 修复action decoding，重写observations | +70 |
| `env/iov_env.py` | 更新obs_dim | 1 |
| **总计** | **3个严重错误已修复** | **~130** |

---

## 验证方法

### 1. 语法检查
```bash
python -m py_compile env/channel.py env/iov_env.py
# ✅ 通过
```

### 2. 运行时验证 (待执行)
```python
env = IoVEnv(config)
obs = env.reset()

# 检查项:
# - obs[i].shape == (2*N_RB + 4,) = (14,)
# - B_k, T_k 从TASK_SIZE, T_MAX逐步递减
# - 每个vehicle选择有效RB (1-N_RB)
```

---

## 论文公式对标总结

| 公式 | 代码位置 | 状态 |
|------|---------|------|
| (1) V2I SINR | - | 纯V2V场景，不适用 |
| (2) RB选择向量 ρ_k | iov_env.py:292 | ✅ 已修复 |
| (3) RB约束 Σρ_k[m]=1 | iov_env.py:292-297 | ✅ 已修复 |
| (4) V2I速率 | - | 纯V2V场景 |
| (5) V2V SINR | channel.py:280 | ✅ 已修复 |
| (6) 干扰计算 | channel.py:283-288 | ✅ 已修复 |
| (7) V2V速率 | channel.py:296 | ✅ 正确 |
| (8) V2I SEE | iov_env.py:352 | ⚠️ 简化版 |
| (10) V2V SEE | iov_env.py:354 | ⚠️ 简化版 |
| (11) 网络SEE | iov_env.py:356-357 | ⚠️ 简化版 |
| (12) 传输速率约束 | iov_env.py:393-395 | ✅ 正确 |
| (13) SINR阈值约束 | config.py:64 | ✅ 正确 |
| (15) 状态信息集合 | iov_env.py:422-429 | ✅ 已修复 |
| (16) 完整状态空间 | iov_env.py:433-446 | ✅ 已修复 |
| (17) 动作空间 | iov_env.py:263-310 | ✅ 已修复 |
| (18) 奖励函数主式 | iov_env.py:369-419 | ✅ 已修复 |
| (19) 从属函数G(x) | iov_env.py:383-388 | ✅ 已修复 |

---

## 下一步工作

### 立即可执行 (Ready Now)
1. ✅ 运行训练验证三项修复是否有效
2. ✅ 检查收敛性和奖励曲线是否改进
3. ✅ 确保B_k/T_k约束正确应用

### 可选优化 (Optional)
1. 改进SEE计算 (公式8, 10, 11) - 添加电路功耗项
2. 添加V2I链路支持 (目前仅V2V)
3. 优化干扰信道矩阵计算的效率

---

## 关键认识 (Key Insights)

1. **干扰信道 vs 自身信道:**
   - 干扰计算必须使用从transmitter到receiver的信道增益
   - 不能混淆为自身信道增益 (transmitter-to-transmitter)

2. **RB约束的强制性:**
   - 论文中Σρ_k[m]=1是硬约束，每个vehicle必须选择恰好一个RB
   - 允许"无RB"选择会违反约束，导致Agent学习歧义

3. **状态空间的完整性:**
   - B_k和T_k是理解传输速率约束(Formula 12)的关键
   - 如果Agent看不到这两个变量，无法学会考虑传输期限
   - 完整干扰向量让Agent针对每个RB的干扰独立决策

4. **观测维度增加的影响:**
   - 从8维→14维
   - DQN网络需要足够的容量处理更多信息
   - 可能需要调整网络大小或学习率

---

**生成时间:** 2026-03-10
**修复状态:** ✅ 三项严重错误已修复，代码通过语法检查
**下一阶段:** 等待训练运行结果验证修复效果
