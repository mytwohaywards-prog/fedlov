"""
DQN Agent for FMDQN Framework (Federated Multi-agent DQN)
Implements Deep Q-Network with experience replay and target network
Supports federated learning with periodic parameter aggregation

Reference:
- DQN paper
- Distributed Resource Allocation With Federated Learning for Delay-Sensitive IoV Services
- Algorithm 1: FMDQN Training Stage with Federated Aggregation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict
from .buffer import ReplayBuffer
from networks.dqn_network import (
    QNetwork, DuelingQNetwork,
    soft_update_dqn, hard_update_dqn,
    get_q_network_parameters, set_q_network_parameters
)


class DQNAgent:
    """
    DQN Agent for FMDQN (Federated Multi-agent DQN) Framework

    Workflow:
    1. Each vehicle (agent) maintains a local Q-network θ_l^k
    2. Agent trains locally on its own experience buffer
    3. Periodically receives global Q-network θ_global from Federal Server
    4. Syncs local network with global parameters
    5. Continues training with updated network

    This enables:
    - Distributed training (each agent trains independently)
    - Knowledge sharing (periodic aggregation of parameters)
    - Faster convergence (global model as reference)
    """

    def __init__(self, agent_id: int, obs_dim: int, n_actions: int, config):
        """
        Initialize DQN agent for FMDQN

        Args:
            agent_id: unique vehicle identifier
            obs_dim: observation dimension
            n_actions: total number of discrete actions
            config: configuration object
        """
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== Local Q-networks (θ_l^k) ==========
        # These are the agent's local networks that will be updated
        self.q_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_network = QNetwork(obs_dim, n_actions).to(self.device)

        # Initialize target network with same parameters
        hard_update_dqn(self.target_q_network, self.q_network)

        # Freeze target network (not directly trained)
        for param in self.target_q_network.parameters():
            param.requires_grad = False

        # ✅ 改: Adam→RMSProp, 严格遵循论文Table 3 (RMSProp with lr=0.001)
        self.optimizer = optim.RMSprop(
            self.q_network.parameters(),
            lr=config.LEARNING_RATE,
            alpha=0.99,
            eps=1e-8
        )

        # Experience replay buffer
        self.buffer = ReplayBuffer(config.BUFFER_SIZE, obs_dim, 1)  # action is single int

        # Exploration parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY

        # Training step counter
        self.train_step = 0
        self.update_counter = 0

        # ========== Federated Learning Tracking ==========
        # Track participation in aggregation rounds
        self.federated_round = 0  # Number of times this agent participated in aggregation
        self.last_aggregation_step = 0

    def choose_action(self, obs: np.ndarray, exploration: bool = True) -> int:
        """
        Select action using epsilon-greedy strategy
        Args:
            obs: observation (obs_dim,)
            exploration: whether to use epsilon-greedy (False = greedy)

        Returns:
            action: discrete action index in [0, n_actions)
        """
        if exploration and np.random.rand() < self.epsilon:
            # Random action for exploration
            action = np.random.randint(0, self.n_actions)
        else:
            # Greedy action: argmax Q(s, a)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax(dim=1).item()

        return action

    def learn(self):
        """
        Update Q-network using DQN algorithm
        - Sample mini-batch from experience replay buffer
        - Compute target Q-values using target network
        - Update Q-network with MSE loss
        - Soft update target network
        """
        if not self.buffer.is_ready(self.config.BATCH_SIZE):
            return

        # Sample mini-batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.buffer.sample_batch(self.config.BATCH_SIZE)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch.astype(int)).squeeze().to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Current Q-values for taken actions
        q_values = self.q_network(obs_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_obs_batch)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
            target_q_values = reward_batch + self.config.GAMMA * (1 - done_batch) * max_next_q_values

        # DQN loss: MSE between current and target Q-values
        dqn_loss = nn.MSELoss()(q_values, target_q_values)

        # Optimization step
        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # ✅ 改: 软更新→硬更新, 严格遵循论文Table 3 (硬更新每8个回合)
        self.train_step += 1
        if self.train_step % self.config.UPDATE_FREQ == 0:
            self.update_counter += 1
            if self.update_counter % self.config.TARGET_UPDATE_FREQ == 0:
                hard_update_dqn(self.target_q_network, self.q_network)  # ✅ 硬更新而不是软更新

        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)

    def store_transition(self, obs: np.ndarray, action: int,
                        reward: float, next_obs: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.buffer.store_transition(obs, np.array([action]), reward, next_obs, done)

    # ========== Local Learning Interface ==========

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Extract Q-network parameters (for knowledge sharing/coordination)
        Returns:
            dict with Q-network parameters
        """
        return get_q_network_parameters(self.q_network)

    def set_parameters(self, params: Dict[str, np.ndarray]):
        """
        Set Q-network parameters (for knowledge sharing/coordination)
        Args:
            params: dict with Q-network parameters
        """
        set_q_network_parameters(self.q_network, params, self.device)
        hard_update_dqn(self.target_q_network, self.q_network)

    def get_buffer_size(self) -> int:
        """Get current buffer occupancy"""
        return len(self.buffer)

    def get_info(self) -> Dict:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'buffer_size': len(self.buffer),
            'epsilon': self.epsilon,
            'train_steps': self.train_step,
            'updates': self.update_counter,
            'federated_rounds': self.federated_round,
        }

    # ========== Federated Learning Interface ==========

    def sync_from_global(self, global_params: Dict[str, torch.Tensor]):
        """
        Sync local Q-network with global model (θ_l^k := θ_global)

        Called by Federal Server after aggregation to ensure all agents
        have the same global model as starting point for next local training

        Args:
            global_params: dict of parameters from global Q-network
        """
        state_dict = self.q_network.state_dict()
        for name, param in global_params.items():
            if name in state_dict:
                state_dict[name] = param.to(self.device)
        self.q_network.load_state_dict(state_dict)

        # Also sync target network to match
        hard_update_dqn(self.target_q_network, self.q_network)

        # Track aggregation
        self.federated_round += 1
        self.last_aggregation_step = self.train_step

    def get_local_params(self) -> Dict[str, torch.Tensor]:
        """
        Get local Q-network parameters for aggregation (θ_l^k)

        Called by Federal Server to collect parameters from all agents
        for aggregation

        Returns:
            dict of local Q-network parameters on CPU
        """
        params = {}
        for name, param in self.q_network.named_parameters():
            params[name] = param.data.cpu().clone()
        return params
