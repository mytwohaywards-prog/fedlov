"""
Distributed Coordinator for HMDQN Agent Communication
Implements inter-agent coordination without central server
Reference: HMDQN decentralized communication mechanism
"""

import numpy as np
from typing import List, Dict, Tuple


class DistributedCoordinator:
    """
    Coordinates multiple DQN agents in a distributed manner
    Enables knowledge sharing and interference avoidance between agents
    """

    def __init__(self, n_vehicles: int, config):
        self.n_vehicles = n_vehicles
        self.config = config
        self.n_rb = config.N_RB  # ✅ 添加: 从config获取N_RB

        # Last actions taken by each vehicle
        self.last_actions = np.zeros(n_vehicles, dtype=int)

        # Communication history (for learning from others' experience)
        self.action_history = [[] for _ in range(n_vehicles)]
        self.reward_history = [[] for _ in range(n_vehicles)]

        # Coordination metrics
        self.channel_conflicts = 0  # Number of times two agents use same RB
        self.power_conflicts = 0     # Number of times power levels cause interference

    def broadcast_actions(self, agents: List) -> Dict:
        """
        Each agent broadcasts its action to others
        Args:
            agents: list of DQNAgent instances

        Returns:
            dict with communication info
        """
        actions = []
        for agent in agents:
            # Get last action from agent's recent experience
            if agent.get_buffer_size() > 0:
                # Greedy action without exploration
                obs_tensor = agent.buffer.obs_buffer[agent.buffer.ptr - 1]
                action = agent.choose_action(obs_tensor, exploration=False)
            else:
                action = np.random.randint(0, agent.n_actions)

            actions.append(action)
            self.last_actions[agent.agent_id] = action

        # Store in history
        for i, agent in enumerate(agents):
            self.action_history[i].append(actions[i])

        return {
            'actions_broadcast': len(actions),
            'timestamp': len(self.action_history[0]),
        }

    def broadcast_rewards(self, rewards: np.ndarray) -> Dict:
        """
        Broadcast reward information between agents for learning
        Args:
            rewards: shape (n_vehicles,) reward for each agent

        Returns:
            dict with reward sharing info
        """
        # Store rewards
        for i, reward in enumerate(rewards):
            self.reward_history[i].append(reward)

        return {
            'mean_reward': float(np.mean(rewards)),
            'max_reward': float(np.max(rewards)),
        }

    def detect_conflicts(self, channel_gains: np.ndarray,
                        power_levels: np.ndarray,
                        allocated_rbs: np.ndarray) -> Dict:
        """
        Detect resource allocation conflicts between agents
        Args:
            channel_gains: shape (n_vehicles, n_rb) channel gains
            power_levels: shape (n_vehicles,) allocated power levels
            allocated_rbs: shape (n_vehicles,) selected RB indices

        Returns:
            dict with conflict information
        """
        conflicts = {
            'rb_conflicts': [],
            'power_conflicts': [],
            'total_conflicts': 0,
        }

        # Detect RB conflicts (same RB used by multiple agents)
        for i in range(self.n_vehicles):
            for j in range(i + 1, self.n_vehicles):
                if allocated_rbs[i] == allocated_rbs[j] and allocated_rbs[i] > 0:
                    conflicts['rb_conflicts'].append((i, j))
                    self.channel_conflicts += 1

        # Detect power conflicts (high power causing interference) (✅ 修复: 添加RB索引边界检查)
        interference = np.zeros(self.n_vehicles)
        for i in range(self.n_vehicles):
            rb_i = int(allocated_rbs[i])
            if rb_i > 0 and rb_i <= self.n_rb:  # 有效的RB
                for j in range(self.n_vehicles):
                    if i != j:
                        rb_j = int(allocated_rbs[j])
                        # 同一RB上的干扰
                        if rb_i == rb_j:
                            interference[i] += power_levels[j] * channel_gains[j, rb_i - 1]  # 注意减1

        for i in range(self.n_vehicles):
            if interference[i] > 0.7:  # High interference threshold
                conflicts['power_conflicts'].append(i)
                self.power_conflicts += 1

        conflicts['total_conflicts'] = len(conflicts['rb_conflicts']) + len(conflicts['power_conflicts'])

        return conflicts

    def suggest_action_coordination(self, agents: List, conflicts: Dict) -> Dict:
        """
        Suggest action adjustments to reduce conflicts
        Args:
            agents: list of DQNAgent instances
            conflicts: conflict information from detect_conflicts()

        Returns:
            dict with coordination suggestions
        """
        suggestions = {}

        # For RB conflicts: suggest different RBs
        for agent_i, agent_j in conflicts['rb_conflicts']:
            if agent_i not in suggestions:
                suggestions[agent_i] = {
                    'type': 'change_rb',
                    'reason': f'conflict_with_agent_{agent_j}',
                    'priority': 'medium',
                }

        # For power conflicts: suggest lower power
        for agent_i in conflicts['power_conflicts']:
            if agent_i not in suggestions:
                suggestions[agent_i] = {
                    'type': 'reduce_power',
                    'reason': 'high_interference',
                    'priority': 'high',
                }

        return suggestions

    def get_communication_overhead(self) -> float:
        """
        Calculate communication overhead for coordination
        Returns:
            overhead: estimated bits per time slot for coordination messages
        """
        # Each agent broadcasts: action (log2(n_actions)), reward (32 bits)
        n_actions = self.config.N_RB_ACTIONS * self.config.N_POWER_ACTIONS * self.config.N_CPU_ACTIONS
        bits_per_agent = np.ceil(np.log2(n_actions)) + 32

        total_overhead = bits_per_agent * self.n_vehicles

        return total_overhead

    def get_stats(self) -> Dict:
        """Get coordinator statistics"""
        reward_means = [np.mean(r) for r in self.reward_history if r]
        avg_reward = float(np.mean(reward_means)) if reward_means else 0.0

        return {
            'total_rb_conflicts': self.channel_conflicts,
            'total_power_conflicts': self.power_conflicts,
            'avg_reward': avg_reward,
            'communication_overhead_bps': self.get_communication_overhead(),
        }

    def reset(self):
        """Reset coordinator state"""
        self.last_actions = np.zeros(self.n_vehicles, dtype=int)
        self.action_history = [[] for _ in range(self.n_vehicles)]
        self.reward_history = [[] for _ in range(self.n_vehicles)]
        self.channel_conflicts = 0
        self.power_conflicts = 0
