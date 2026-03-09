"""
DQN (Deep Q-Network) for HMDQN Framework
Reference: DQN from "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
and HMDQN from paper Fig.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """
    Deep Q-Network: obs_dim -> [256, 128, 64] -> n_actions
    Outputs Q-values for all discrete actions
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network
        Args:
            obs: shape (batch_size, obs_dim)
        Returns:
            q_values: shape (batch_size, n_actions)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.output(x)

        return q_values


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN: Separates value and advantage streams
    V(s) + A(s,a) - mean(A) to improve learning stability
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super(DuelingQNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)

        # Value stream
        self.value_fc = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)

        # Advantage stream
        self.adv_fc = nn.Linear(128, 64)
        self.advantage = nn.Linear(64, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling Q-network
        Args:
            obs: shape (batch_size, obs_dim)
        Returns:
            q_values: shape (batch_size, n_actions)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value(value)

        # Advantage stream
        adv = F.relu(self.adv_fc(x))
        adv = self.advantage(adv)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = value + (adv - adv.mean(dim=1, keepdim=True))

        return q_values


def soft_update_dqn(target_model: nn.Module, source_model: nn.Module, tau: float):
    """
    Soft update for DQN target network: target = (1-tau)*target + tau*source
    Args:
        target_model: target Q-network
        source_model: learned Q-network
        tau: update coefficient (typically 0.01)
    """
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)


def hard_update_dqn(target_model: nn.Module, source_model: nn.Module):
    """Hard update: copy all parameters from source to target"""
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(source_param.data)


def get_q_network_parameters(model: nn.Module) -> dict:
    """Extract all parameters from a Q-network as a dictionary"""
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.clone().detach().cpu().numpy()
    return params


def set_q_network_parameters(model: nn.Module, params: dict, device: str = 'cpu'):
    """Set Q-network parameters from a dictionary"""
    for name, param in model.named_parameters():
        if name in params:
            param.data = torch.from_numpy(params[name]).to(device).float()


def initialize_weights(model: nn.Module):
    """Initialize network weights uniformly"""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -3e-3, 3e-3)
