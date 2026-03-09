"""
TD3 (Twin Delayed DDPG) Agent with Federated Learning Interface
Reference: TD3_agent.py from V2X-RRM-IEEE-OJ-COMS-2024-main
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict
from .buffer import ReplayBuffer
from networks.networks import (
    ActorNetwork, DualCriticNetwork,
    soft_update, hard_update, initialize_weights,
    get_network_parameters, set_network_parameters
)


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent for multi-vehicle resource allocation
    Implements federated learning interface for parameter synchronization
    """

    def __init__(self, agent_id: int, obs_dim: int, act_dim: int, config):
        """
        Initialize TD3 agent
        Args:
            agent_id: unique identifier
            obs_dim: observation dimension
            act_dim: action dimension
            config: configuration object
        """
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks: actor, critic1, critic2 and their targets
        self.actor = ActorNetwork(obs_dim, act_dim).to(self.device)
        self.target_actor = ActorNetwork(obs_dim, act_dim).to(self.device)

        self.critic = DualCriticNetwork(obs_dim, act_dim).to(self.device)
        self.target_critic = DualCriticNetwork(obs_dim, act_dim).to(self.device)

        # Initialize target networks with same parameters
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # Freeze target networks (not directly trained)
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                         lr=config.LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                          lr=config.LEARNING_RATE_CRITIC)

        # Experience replay buffer
        self.buffer = ReplayBuffer(config.BUFFER_SIZE, obs_dim, act_dim)

        # TD3 specific: delayed policy update counter
        self.policy_delay_counter = 0
        self.policy_delay = config.POLICY_DELAY

        # Exploration noise
        self.exploration_std = 0.1

    def choose_action(self, obs: np.ndarray, noise: bool = True) -> np.ndarray:
        """
        Select action using actor network with optional exploration noise
        Args:
            obs: observation (obs_dim,)
            noise: whether to add Gaussian noise for exploration

        Returns:
            action: shape (act_dim,) in [0, 1]
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(obs_tensor).squeeze(0).cpu().numpy()

        if noise:
            noise_vec = np.random.normal(0, self.exploration_std, self.act_dim)
            action = action + noise_vec
            action = np.clip(action, 0, 1)

        return action

    def learn(self):
        """
        Update networks using TD3 algorithm
        - Update critic networks (both Q1 and Q2) using Bellman equation
        - Periodically update actor network and target networks
        """
        if not self.buffer.is_ready(self.config.BATCH_SIZE):
            return

        # Sample mini-batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.buffer.sample_batch(self.config.BATCH_SIZE)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # ========== Critic Update ==========
        with torch.no_grad():
            # Target action with smoothing noise
            target_action = self.target_actor(next_obs_batch)
            noise = torch.randn_like(target_action) * self.config.TARGET_NOISE
            noise = torch.clamp(noise, -self.config.TARGET_NOISE_CLIP, self.config.TARGET_NOISE_CLIP)
            target_action = torch.clamp(target_action + noise, 0, 1)

            # Double Q-learning: use minimum of two Q-functions
            target_q1, target_q2 = self.target_critic(next_obs_batch, target_action)
            target_q = torch.min(target_q1, target_q2)

            # Bellman backup: r + gamma * (1 - done) * Q_target
            target_q_value = reward_batch + self.config.GAMMA * (1 - done_batch) * target_q

        # Critic loss: MSE of both Q-functions
        q1, q2 = self.critic(obs_batch, action_batch)
        critic_loss = nn.MSELoss()(q1, target_q_value) + nn.MSELoss()(q2, target_q_value)

        # Critic optimization step
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== Actor Update (Delayed) ==========
        self.policy_delay_counter += 1

        if self.policy_delay_counter % self.policy_delay == 0:
            # Actor loss: negative mean Q1
            actor_action = self.actor(obs_batch)
            actor_loss = -self.critic.critic1(obs_batch, actor_action).mean()

            # Actor optimization step
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ========== Target Network Update (Soft Update) ==========
            soft_update(self.target_actor, self.actor, self.config.TAU)
            soft_update(self.target_critic, self.critic, self.config.TAU)

    # ========== Federated Learning Interface ==========

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Extract all learnable parameters for federated averaging
        Returns:
            dict with keys: 'actor', 'critic1', 'critic2'
        """
        params = {
            'actor': get_network_parameters(self.actor),
            'critic1': get_network_parameters(self.critic.critic1),
            'critic2': get_network_parameters(self.critic.critic2),
        }
        return params

    def set_parameters(self, global_params: Dict[str, Dict[str, np.ndarray]]):
        """
        Set global parameters from federated server
        Updates both current and target networks

        Args:
            global_params: dict with keys 'actor', 'critic1', 'critic2'
        """
        set_network_parameters(self.actor, global_params['actor'], self.device)
        set_network_parameters(self.critic.critic1, global_params['critic1'], self.device)
        set_network_parameters(self.critic.critic2, global_params['critic2'], self.device)

        # Synchronize target networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def store_transition(self, obs: np.ndarray, action: np.ndarray,
                        reward: float, next_obs: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.buffer.store_transition(obs, action, reward, next_obs, done)

    def get_buffer_size(self) -> int:
        """Get current buffer occupancy"""
        return len(self.buffer)

    def get_info(self) -> Dict:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'buffer_size': len(self.buffer),
            'exploration_std': self.exploration_std,
            'policy_delay_counter': self.policy_delay_counter,
        }
