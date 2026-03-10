"""
Federal Server for FMDQN (Federated Multi-agent DQN)
Manages global Q-network and coordinates parameter aggregation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
import copy
from networks.dqn_network import QNetwork, hard_update_dqn


class FederalServer:
    """
    Central server that manages federated learning for multi-agent DQN

    Workflow:
    1. Each agent trains locally on its own Q-network (θ_l^k)
    2. Server periodically collects parameters from all agents
    3. Server computes weighted average: θ_global = Σ w_k * θ_l^k
    4. Server distributes θ_global back to all agents
    5. Agents update their local Q-networks with θ_global
    """

    def __init__(self, obs_dim: int, n_actions: int, n_agents: int, config):
        """
        Initialize Federal Server

        Args:
            obs_dim: observation dimension
            n_actions: number of actions
            n_agents: total number of agents
            config: configuration object
        """
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Global Q-network (shared reference model)
        self.global_q_network = QNetwork(obs_dim, n_actions).to(self.device)

        # Aggregation parameters
        self.aggregation_step = 0
        self.aggregation_history = []  # Track aggregation events

        # Agent references (will be set after agents are created)
        self.agents = []

        # Aggregation frequency (from config or default)
        self.aggregation_freq = getattr(config, 'AGGREGATION_FREQ', 50)  # Default: every 50 episodes

        # Aggregation weights (equal by default: 1/K for each agent)
        self.agent_weights = np.ones(n_agents) / n_agents

    def set_agents(self, agents: List):
        """Register agents for aggregation"""
        self.agents = agents
        assert len(agents) == self.n_agents, "Number of agents mismatch"

    def aggregate(self):
        """
        执行联邦聚合 (Federated Aggregation)

        步骤:
        1. 收集所有agents的本地Q网络参数
        2. 计算加权平均: θ_global = Σ w_k * θ_l^k
        3. 将θ_global分发回所有agents
        4. 所有agents同步其local networks

        返回:
            aggregation_info: dict with aggregation statistics
        """
        if not self.agents:
            return None

        self.aggregation_step += 1

        # Step 1: 收集所有agents的参数
        agent_params = []
        for agent in self.agents:
            params = self._extract_parameters(agent.q_network)
            agent_params.append(params)

        # Step 2: 计算加权平均
        global_params = self._compute_weighted_average(agent_params)

        # Step 3: 更新全局模型
        self._set_parameters(self.global_q_network, global_params)

        # Step 4: 分发到所有agents并让它们同步
        for i, agent in enumerate(self.agents):
            self._set_parameters(agent.q_network, global_params)
            # 同时更新target network
            hard_update_dqn(agent.target_q_network, agent.q_network)

        # 记录统计信息
        info = {
            'aggregation_step': self.aggregation_step,
            'n_agents': self.n_agents,
            'timestamp': None,
        }
        self.aggregation_history.append(info)

        return info

    def _extract_parameters(self, network: nn.Module) -> Dict[str, torch.Tensor]:
        """
        从网络提取所有参数

        Args:
            network: PyTorch network model

        Returns:
            params: dict of parameter names -> values (CPU)
        """
        params = {}
        for name, param in network.named_parameters():
            params[name] = param.data.cpu().clone()
        return params

    def _compute_weighted_average(self, agent_params_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算所有agents参数的加权平均

        θ_global = Σ w_k * θ_l^k

        Args:
            agent_params_list: list of parameter dicts from each agent

        Returns:
            averaged_params: dict of averaged parameters
        """
        assert len(agent_params_list) == self.n_agents, "Number of parameter sets mismatch"

        averaged_params = {}

        # 获取第一个agent的参数名作为template
        template_params = agent_params_list[0]

        for param_name in template_params.keys():
            # 初始化累加器
            param_sum = torch.zeros_like(template_params[param_name])

            # 累加加权参数
            for k, agent_params in enumerate(agent_params_list):
                w_k = self.agent_weights[k]  # agent k的权重
                param_sum += w_k * agent_params[param_name]

            averaged_params[param_name] = param_sum

        return averaged_params

    def _set_parameters(self, network: nn.Module, params: Dict[str, torch.Tensor]):
        """
        将参数dict设置到网络

        Args:
            network: PyTorch network model
            params: dict of parameter names -> values
        """
        state_dict = network.state_dict()
        for name, param in params.items():
            state_dict[name] = param.to(self.device)
        network.load_state_dict(state_dict)

    def get_global_weights(self) -> Dict[str, np.ndarray]:
        """获取全局模型的权重（用于评估或保存）"""
        weights = {}
        for name, param in self.global_q_network.named_parameters():
            weights[name] = param.data.cpu().numpy()
        return weights

    def get_aggregation_statistics(self) -> Dict:
        """获取聚合统计信息"""
        return {
            'total_aggregations': self.aggregation_step,
            'aggregation_freq': self.aggregation_freq,
            'n_agents': self.n_agents,
            'history_length': len(self.aggregation_history),
        }
