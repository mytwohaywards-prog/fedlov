"""
Actor and Critic Networks for TD3 Algorithm
Reference: TD3_Networks.py from V2X-RRM-IEEE-OJ-COMS-2024-main
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    """
    Actor Network: obs_dim -> [256, 128, 64] -> act_dim
    Output: Sigmoid to keep actions in [0, 1]
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network
        Args:
            obs: shape (batch_size, obs_dim)
        Returns:
            actions: shape (batch_size, act_dim) in [0, 1]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = torch.sigmoid(self.output(x))  # Sigmoid for [0, 1]

        return actions


class CriticNetwork(nn.Module):
    """
    Critic Network (Q-function): (obs_dim + act_dim) -> [256, 128, 64] -> 1
    Takes state and action as input, outputs Q-value
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()

        input_dim = obs_dim + act_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network
        Args:
            obs: shape (batch_size, obs_dim)
            actions: shape (batch_size, act_dim)
        Returns:
            q_value: shape (batch_size, 1)
        """
        x = torch.cat([obs, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.output(x)

        return q_value


class DualCriticNetwork(nn.Module):
    """
    Dual Critic Networks for TD3 (Q1 and Q2)
    Helps reduce overestimation bias
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super(DualCriticNetwork, self).__init__()

        self.critic1 = CriticNetwork(obs_dim, act_dim, hidden_dim)
        self.critic2 = CriticNetwork(obs_dim, act_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        """
        Forward pass through both critics
        Returns:
            q1, q2: both shape (batch_size, 1)
        """
        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        return q1, q2

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Only use critic 1 for inference (faster)"""
        return self.critic1(obs, actions)


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic module for easier parameter management
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super(ActorCriticNetwork, self).__init__()

        self.actor = ActorNetwork(obs_dim, act_dim, hidden_dim)
        self.critic = DualCriticNetwork(obs_dim, act_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor = None):
        """
        Args:
            obs: shape (batch_size, obs_dim)
            actions: if None, only return actor output; else return actor and critic outputs
        """
        policy = self.actor(obs)

        if actions is None:
            return policy

        q1, q2 = self.critic(obs, actions)
        return policy, q1, q2


def soft_update(target_model: nn.Module, source_model: nn.Module, tau: float):
    """
    Soft update (polyak averaging): target = (1-tau)*target + tau*source
    Args:
        target_model: target network
        source_model: source (learned) network
        tau: update coefficient (typically 0.01)
    """
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)


def hard_update(target_model: nn.Module, source_model: nn.Module):
    """Hard update: copy all parameters from source to target"""
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(source_param.data)


def initialize_weights(model: nn.Module):
    """Initialize network weights with uniform distribution"""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -3e-3, 3e-3)


def get_network_parameters(model: nn.Module) -> dict:
    """Extract all parameters from a network as a dictionary"""
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.clone().detach().cpu().numpy()
    return params


def set_network_parameters(model: nn.Module, params: dict, device: str = 'cpu'):
    """Set network parameters from a dictionary"""
    for name, param in model.named_parameters():
        if name in params:
            param.data = torch.from_numpy(params[name]).to(device).float()
