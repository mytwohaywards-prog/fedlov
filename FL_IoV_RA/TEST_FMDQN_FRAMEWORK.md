# FMDQN框架快速验证方案

## 当前问题

200 episodes训练后：
- 奖励仍在0-30范围波动（应该上升到95）
- 延迟7-9秒（约束2秒，理论最优1秒）
- 成功率0-30%（应该上升到95%）

**根本原因**：系统参数设置过于严格，agents难以学会协调

## 快速验证方案

### 方案1: 简化测试（最快，10分钟）

**目标**：验证FMDQN聚合机制是否工作，与具体系统参数无关

**改动**：
1. 减少车辆数：10 → 3 (减少竞争复杂度)
2. 增加资源：
   - BW_PER_RB: 0.72MHz → 5MHz (足够带宽)
   - CPU: 100MHz → 500MHz (足够计算)
3. 简化任务：
   - TASK_SIZE: 16960bits → 1000bits
   - CPU_CYCLES: 100M → 10M
4. 放宽约束：
   - T_MAX: 2s → 10s (容易满足)
   - SINR_THRESHOLD: 0dBm (保留)

**预期**：
- 初期奖励 → 50-70 (足够的任务完成)
- 应该看到 **清晰的上升趋势** (30→95 无法实现，但至少应该上升)
- 如果上升 → FMDQN框架OK ✓
- 如果不上升 → 框架有问题 ✗

### 方案2: 理想信道测试（5分钟）

**最极端的简化**：移除无线信道模型的随机性

```python
# channel.py 修改
def _update_path_loss(self):
    # 使用固定的理想信道增益
    self.path_loss = np.ones((self.n_vehicles, self.n_rb)) * (-50)  # 固定-50dB

def _update_shadowing_initial(self):
    self.shadowing = np.zeros((self.n_vehicles, self.n_rb))

def _update_fading(self):
    self.fading = np.ones((self.n_vehicles, self.n_rb))
```

这样SINR会非常高，agents可以专注于学习资源分配的协调。

### 方案3: 完整验证（1小时）

同时应用方案1的所有参数修改，运行200 episodes看收敛曲线。

## 推荐执行顺序

1. **先做方案2** (5分钟) - 理想信道
   - 最快速的验证
   - 如果这都不能改进，说明问题更深层

2. **如果方案2有效** → 再做方案1
   - 逐步增加难度
   - 确定哪个参数导致无法学习

3. **收集数据后** → 调整论文参数
   - 基于验证结果，决定是否需要修改论文参数

## 当前结论

❌ **FMDQN框架本身的工作状态未知**（被系统参数困难掩盖）

需要用简化场景来验证。
