"""
Federated Learning Server with FedAvg Aggregation
Reference: TD3_server.py from V2X-RRM-IEEE-OJ-COMS-2024-main
"""

import numpy as np
from typing import List, Dict
from networks.networks import ActorNetwork, DualCriticNetwork, hard_update, set_network_parameters


class FLServer:
    """
    Federated Learning Server
    Aggregates model parameters from multiple agents using FedAvg
    """

    def __init__(self, obs_dim: int, act_dim: int, config):
        """
        Initialize FL server
        Args:
            obs_dim: observation dimension
            act_dim: action dimension
            config: configuration object
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config

        # Global networks
        self.global_actor = ActorNetwork(obs_dim, act_dim)
        self.global_critic = DualCriticNetwork(obs_dim, act_dim)

        # Aggregation statistics
        self.round_count = 0
        self.aggregation_history = []

    def broadcast(self, agents: List) -> Dict:
        """
        Send global model to all agents
        Args:
            agents: list of TD3Agent instances

        Returns:
            dict with broadcast info
        """
        # Get global parameters
        global_params = {
            'actor': self._get_weights_dict(self.global_actor),
            'critic1': self._get_weights_dict(self.global_critic.critic1),
            'critic2': self._get_weights_dict(self.global_critic.critic2),
        }

        # Send to all agents
        for agent in agents:
            agent.set_parameters(global_params)

        return {'broadcast_to': len(agents), 'round': self.round_count}

    def aggregate(self, agents: List, participation_rate: float = 1.0) -> Dict:
        """
        Aggregate parameters from agents using FedAvg
        Args:
            agents: list of TD3Agent instances
            participation_rate: fraction of agents to include (0.0 - 1.0)

        Returns:
            dict with aggregation statistics
        """
        # Determine participating agents
        n_participants = max(1, int(len(agents) * participation_rate))
        if n_participants < len(agents):
            selected_indices = np.random.choice(len(agents), n_participants, replace=False)
            participating_agents = [agents[i] for i in selected_indices]
        else:
            participating_agents = agents

        # Collect parameters from all participating agents
        agent_params_list = []
        for agent in participating_agents:
            params = agent.get_parameters()
            agent_params_list.append(params)

        # FedAvg: simple averaging
        aggregated_params = self._fedavg(agent_params_list)

        # Update global networks
        self._set_global_params(aggregated_params)

        # Record statistics
        self.round_count += 1
        self.aggregation_history.append({
            'round': self.round_count,
            'participants': len(participating_agents),
            'participation_rate': len(participating_agents) / len(agents),
        })

        return {
            'round': self.round_count,
            'participants': len(participating_agents),
            'participation_rate': participation_rate,
        }

    def _fedavg(self, agent_params_list: List[Dict]) -> Dict:
        """
        Federated Averaging: average all agent parameters
        Args:
            agent_params_list: list of dicts with keys 'actor', 'critic1', 'critic2'

        Returns:
            aggregated_params: dict with same structure as input
        """
        n_agents = len(agent_params_list)

        aggregated = {
            'actor': {},
            'critic1': {},
            'critic2': {},
        }

        # Average actor parameters
        for param_name in agent_params_list[0]['actor'].keys():
            param_list = [params['actor'][param_name] for params in agent_params_list]
            aggregated['actor'][param_name] = np.mean(param_list, axis=0)

        # Average critic1 parameters
        for param_name in agent_params_list[0]['critic1'].keys():
            param_list = [params['critic1'][param_name] for params in agent_params_list]
            aggregated['critic1'][param_name] = np.mean(param_list, axis=0)

        # Average critic2 parameters
        for param_name in agent_params_list[0]['critic2'].keys():
            param_list = [params['critic2'][param_name] for params in agent_params_list]
            aggregated['critic2'][param_name] = np.mean(param_list, axis=0)

        return aggregated

    def _get_weights_dict(self, network):
        """Extract weights from a network as dictionary"""
        weights = {}
        for name, param in network.named_parameters():
            weights[name] = param.data.cpu().numpy().copy()
        return weights

    def _set_global_params(self, aggregated_params: Dict):
        """Set global networks with aggregated parameters"""
        set_network_parameters(self.global_actor, aggregated_params['actor'])
        set_network_parameters(self.global_critic.critic1, aggregated_params['critic1'])
        set_network_parameters(self.global_critic.critic2, aggregated_params['critic2'])

    def get_global_params(self) -> Dict:
        """Get current global parameters"""
        return {
            'actor': self._get_weights_dict(self.global_actor),
            'critic1': self._get_weights_dict(self.global_critic.critic1),
            'critic2': self._get_weights_dict(self.global_critic.critic2),
        }

    def get_history(self) -> List[Dict]:
        """Get aggregation history"""
        return self.aggregation_history

    def save_checkpoint(self, filepath: str):
        """Save server state to disk"""
        import torch
        checkpoint = {
            'round': self.round_count,
            'actor_state': self.global_actor.state_dict(),
            'critic_state': self.global_critic.state_dict(),
            'history': self.aggregation_history,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load server state from disk"""
        import torch
        checkpoint = torch.load(filepath)
        self.round_count = checkpoint['round']
        self.global_actor.load_state_dict(checkpoint['actor_state'])
        self.global_critic.load_state_dict(checkpoint['critic_state'])
        self.aggregation_history = checkpoint['history']
