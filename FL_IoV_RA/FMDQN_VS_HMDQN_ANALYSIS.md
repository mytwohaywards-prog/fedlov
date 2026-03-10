# 论文FMDQN vs 我们的HMDQN核心差异分析

## 第一部分：论文提出的FMDQN（Algorithm 1）

### 核心特征：Federated Aggregation

```
Algorithm 1: FMDQN - Training Stage

1. Initialization: 启动模拟器和DQN参数θ_l
2. For each episode e do:
   3. Update VUEs位置和slow fading
   4. For each time step t do:
      5. For each agent k do:
         6. Observe state s_t^k
         7. Select action a_t^k based on max Q(s_t^k, a; θ_l)
         8. Transmit and receive
         9. Observe reward r_t^k and next state s_{t+1}^k
         10. Store transition in replay buffer
      11. End for each agent
      12. Every C steps: Local training (gradient descent)
   13. End for each time step
14. Every F episodes: Federated Aggregation
    - Collect θ_l from all agents
    - Compute θ_global = Σ(w_k * θ_l^k)  # weighted average
    - Distribute θ_global back to all agents
15. End for each episode

Output: Network SEE and V2V success rate
```

### 关键部分：Federated Aggregation Algorithm Design

```
每个agent独立训练自己的Q网络θ_l

定期聚合步骤（论文第III.B部分）:
1. 所有agents计算各自的梯度∇L(θ_l)
2. 梯度汇聚到中央服务器
3. 服务器计算加权平均：θ_global = Σ w_k * θ_l^k
4. 将θ_global分发回所有agents
5. 每个agent更新：θ_l := θ_global

这样可以：
- 加速全局收敛
- 共享知识
- 保持分散控制
```

---

## 第二部分：我们实现的HMDQN问题

### 当前架构

```python
# 我们的做法：
- 每个agent有独立的Q网络
- 没有中央全局模型
- 通过coordinator检测冲突
- 完全独立学习
```

### 问题

```
1. 没有联邦聚合机制
2. 没有全局模型同步
3. 无法共享学习经验
4. 各agent收敛速度不一
5. 导致初期奖励不稳定
```

---

## 第三部分：FMDQN的核心算法参数

从论文的Algorithm 1和Table II:

### 训练参数

| 参数 | 论文值 | 含义 |
|------|-------|------|
| C | ? | 本地训练频率（每C步训练一次） |
| F | ? | 聚合频率（每F个episode聚合一次） |
| w_k | 等权重 | 第k个agent的聚合权重（通常1/K） |
| θ_global | - | 全局Q网络参数 |
| θ_l^k | - | agent k的本地Q网络参数 |

### 缺失信息

论文没有明确说明：
- C的值（我们假设=4，每4步后更新）
- F的值（我们假设=50或FL_ROUNDS）
- 聚合权重的选择方式

---

## 第四部分：为什么我们的性能不好

### 原因链条

```
1. 我们没有全局模型
   ↓
2. 各agent独立从随机初始状态学习
   ↓
3. 初期成功率很低（SINR随机太高/太低）
   ↓
4. 初期奖励波动大
   ↓
5. 无法集中学习全局最优策略
   ↓
6. 收敛速度慢，稳定性差
   ↓
7. 曲线不如论文平滑
```

### 对比论文

```
论文FMDQN:
1. 有全局模型作为参考
2. 定期同步所有agents
3. 共享学习经验
4. 快速收敛到全局最优
5. 曲线平滑上升 (30→95)
```

---

## 第五部分：修正方案

### 需要实现的核心改动

#### 1. 全局Q网络（新增）
```python
class FederatedDQNAgent:
    def __init__(self):
        self.local_q_network = DQN(...)  # 本地网络
        self.local_target_network = DQN(...)
        # 不再有本地参数汇聚到全局

    def receive_global_params(self, global_theta):
        """接收全局模型参数"""
        self.local_q_network.load_state_dict(global_theta)
```

#### 2. 中央联邦服务器（新增）
```python
class FederalServer:
    def __init__(self, n_agents, q_network_template):
        self.global_q_network = q_network_template()  # 全局模型
        self.agents = ...

    def aggregate(self):
        """聚合所有agents的参数"""
        global_params = {}
        for name, param in self.global_q_network.named_parameters():
            # 计算加权平均
            param_sum = torch.zeros_like(param)
            for agent in self.agents:
                w_k = 1.0 / len(self.agents)  # 等权重
                param_sum += w_k * agent.local_q_network.state_dict()[name]
            global_params[name] = param_sum

        # 更新全局模型
        self.global_q_network.load_state_dict(global_params)

        # 分发到所有agents
        for agent in self.agents:
            agent.receive_global_params(global_params)
```

#### 3. 训练循环修改
```python
# 当前（错误）:
for episode in range(NUM_EPISODES):
    for agent in agents:
        agent.train()  # 完全独立

# 应该改为（正确的FMDQN）:
for episode in range(NUM_EPISODES):
    for agent in agents:
        agent.local_train()  # 本地训练

    if episode % F == 0:  # 每F个episode
        federal_server.aggregate()  # 全局聚合和同步
```

---

## 第六部分：预期修正效果

### 修正前 vs 修正后

| 指标 | 修正前 | 修正后 |
|------|-------|--------|
| 初期奖励 | 78+ (太高) | 30+ (正确) ✓ |
| 奖励平滑性 | 波动大 | 平滑上升 ✓ |
| 收敛速度 | 慢 | 快 ✓ |
| 最终奖励 | 83-87 | 90-95 ✓ |
| 曲线形状 | 反向 | 与论文一致 ✓ |

---

## 第七部分：代码改造步骤

### Step 1: 提取共享全局Q网络
```python
# 从 agents/dqn_agent.py 中提取
# 创建一个共享的全局Q网络模板
global_q_network = DQNNetwork(obs_dim, action_space)
```

### Step 2: 修改Agent为本地模式
```python
class DQNAgent:
    def __init__(self, local_id, global_network):
        self.local_q = copy.deepcopy(global_network)
        self.local_target = copy.deepcopy(global_network)
        self.global_network = global_network  # 引用
```

### Step 3: 添加Federal Server
```python
class FederalServer:
    def __init__(self, agents, global_network):
        self.agents = agents
        self.global_network = global_network

    def aggregate_and_sync(self):
        """执行联邦聚合"""
        # ... 聚合逻辑
```

### Step 4: 修改main.py的训练循环
```python
# 添加聚合检查点
if episode % AGGREGATION_FREQ == 0:
    federal_server.aggregate_and_sync()
```

---

## 第八部分：关键参数确定

### 待确认参数（从论文Table II提取缺失的）

1. **聚合频率 F**:
   - 论文未明确
   - 建议: F = 50 (每50个episode聚合一次)

2. **聚合权重 w_k**:
   - 论文：等权重 (1/K)
   - 实现：所有agents权重相同

3. **本地训练频率 C**:
   - 论文未明确
   - 建议：每个time step执行梯度步 (C=1)

---

## 总结：FMDQN vs HMDQN

| 特性 | FMDQN (论文) | HMDQN (我们) |
|------|-------------|-----------|
| 全局模型 | ✓ 有 | ✗ 无 |
| 联邦聚合 | ✓ 有 | ✗ 无 |
| 参数同步 | ✓ 定期 | ✗ 无 |
| 共享学习 | ✓ 是 | ✗ 否 |
| 初期收敛 | ✓ 快 | ✗ 慢 |
| 稳定性 | ✓ 高 | ✗ 低 |
| **性能** | **30→95** | **78→87** |
