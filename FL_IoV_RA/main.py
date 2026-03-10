"""
Main training and evaluation script for FMDQN-based IoV Resource Allocation
Strictly follows the IEEE paper: "Distributed Resource Allocation With Federated Learning
for Delay-Sensitive IoV Services" - Algorithm 1: FMDQN Training Stage

FMDQN = Federated Multi-agent DQN with periodic parameter aggregation

Usage:
    python main.py --mode train       # Train FMDQN model
    python main.py --mode eval        # Evaluate all methods
"""

import argparse
import numpy as np
import torch
from typing import List, Dict
import matplotlib.pyplot as plt

# Import modules
import config
from env import IoVEnv
from agents.dqn_agent import DQNAgent
from agents.federal_server import FederalServer
from agents.coordinator import DistributedCoordinator
from baselines import RandomAllocation, EqualAllocation, GreedyAllocation
from utils import (
    create_directories, save_results, plot_convergence, plot_comparison,
    plot_delay_distribution, plot_energy_distribution, print_results, get_time_str
)


class FMDQNTrainer:
    """
    Trainer for Federated Multi-agent DQN (FMDQN)

    Implements Algorithm 1: FMDQN Training Stage from the IEEE paper
    - Each vehicle maintains a local Q-network (θ_l^k)
    - Local training on experience replay buffer
    - Periodic federated aggregation of parameters
    - Synchronized global model distribution
    """

    def __init__(self, cfg):
        self.config = cfg
        self.env = IoVEnv(cfg)

        # Initialize DQN agents (one per vehicle)
        self.agents = [DQNAgent(i, self.env.obs_dim, self.env.n_actions, cfg)
                       for i in range(cfg.N_VEHICLES)]

        # ========== FMDQN: Federal Server ==========
        # Central server that manages global Q-network and aggregation
        self.federal_server = FederalServer(
            obs_dim=self.env.obs_dim,
            n_actions=self.env.n_actions,
            n_agents=cfg.N_VEHICLES,
            config=cfg
        )
        # Register agents with server
        self.federal_server.set_agents(self.agents)

        # Distributed coordinator
        self.coordinator = DistributedCoordinator(cfg.N_VEHICLES, cfg)

        # Training statistics
        self.episode_rewards = []
        self.episode_delays = []
        self.episode_energies = []
        self.episode_constraint_satisfaction = []

        print(f"[FMDQN Trainer] Initialized with {cfg.N_VEHICLES} federated DQN agents")
        print(f"  Obs dim: {self.env.obs_dim}, Action space: {self.env.n_actions} discrete actions")
        print(f"  Discrete action decomposition: RB={cfg.N_RB_ACTIONS}, Power={cfg.N_POWER_ACTIONS}, CPU={cfg.N_CPU_ACTIONS}")
        print(f"  Federated Aggregation enabled: every {self.federal_server.aggregation_freq} episodes")

    def train(self, num_episodes: int = None):
        """
        FMDQN Training Loop (Algorithm 1)

        Workflow:
        1. Each episode: All agents collect local experience
        2. All agents perform local training (gradient steps)
        3. Every F episodes: Execute Federated Aggregation
           - Collect local parameters from all agents
           - Compute global parameter average
           - Sync all agents with global model
        4. Repeat until convergence

        Args:
            num_episodes: number of training episodes (default: config.FL_ROUNDS)
        """
        if num_episodes is None:
            num_episodes = self.config.FL_ROUNDS

        print(f"\n[Training] Starting {num_episodes} episodes of FMDQN with federated aggregation...")

        for episode in range(num_episodes):
            # Reset environment
            obs_list = self.env.reset()
            episode_reward = 0
            episode_step = 0

            for step in range(self.config.LOCAL_STEPS):
                # 1. Each agent selects action independently (epsilon-greedy)
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.choose_action(obs_list[i], exploration=True)
                    actions.append(action)

                actions = np.array(actions)

                # 2. Environment step
                next_obs_list, rewards, done, info = self.env.step(actions)

                # 3. Store experiences and learn
                for i, agent in enumerate(self.agents):
                    agent.store_transition(
                        obs_list[i], actions[i], rewards[i], next_obs_list[i], done
                    )
                    agent.learn()

                # 4. Distributed coordination: detect and communicate conflicts (✅ 修复: 改用step返回的信息)
                if step % 5 == 0:  # Coordinate every 5 steps
                    channel_gains = self.env.channel.get_channel_gain()
                    # 使用step中已包含的资源分配信息
                    allocated_rbs = np.array([
                        (a // (self.config.N_POWER_ACTIONS * self.config.N_CPU_ACTIONS))
                        for a in actions
                    ])
                    power_allocation = np.array(info['power_levels']) if 'power_levels' in info else np.ones(self.config.N_VEHICLES)

                    conflicts = self.coordinator.detect_conflicts(
                        channel_gains, power_allocation, allocated_rbs
                    )

                    # Optional: Log conflicts
                    if conflicts['total_conflicts'] > 0 and self.config.VERBOSE:
                        print(f"  Episode {episode}, Step {step}: {conflicts['total_conflicts']} conflicts detected")

                episode_reward += np.mean(rewards)
                obs_list = next_obs_list
                episode_step = step

            # Collect episode statistics
            stats = self.env.get_episode_stats()
            self.episode_rewards.append(episode_reward / (episode_step + 1))
            self.episode_delays.append(stats['mean_delay'])
            self.episode_energies.append(stats['mean_energy'])
            self.episode_constraint_satisfaction.append(stats['constraint_satisfaction_rate'])

            # ✅ Epsilon衰减 (论文Table II: 前800个episode线性衰减)
            for agent in self.agents:
                agent.epsilon = max(agent.epsilon_end,
                                   agent.epsilon - (1.0 - agent.epsilon_end) / self.config.EPSILON_DECAY)

            # ========== FMDQN: Federated Aggregation ==========
            # 每F个episodes执行一次聚合 (Algorithm 1 Line 14)
            if (episode + 1) % self.federal_server.aggregation_freq == 0:
                aggregation_info = self.federal_server.aggregate()
                if self.config.VERBOSE:
                    print(f"[Federated Aggregation] Episode {episode + 1}: "
                          f"Aggregation round {aggregation_info['aggregation_step']} completed")

            # Logging
            if (episode + 1) % self.config.LOG_INTERVAL == 0:
                print(f"[Episode {episode + 1:3d}] "
                      f"Reward: {self.episode_rewards[-1]:7.3f}, "
                      f"Delay: {stats['mean_delay']:7.3f}s, "
                      f"Energy: {stats['mean_energy']:7.3f}J, "
                      f"CSR: {stats['constraint_satisfaction_rate']:.2%}")

                # Print agent epsilon values
                epsilons = [agent.epsilon for agent in self.agents]
                print(f"         Epsilon range: [{min(epsilons):.4f}, {max(epsilons):.4f}], "
                      f"Buffer sizes: {[agent.get_buffer_size() for agent in self.agents[:3]]}...")

        print("[Training] Completed!")

        # Print federated learning statistics
        fed_stats = self.federal_server.get_aggregation_statistics()
        print(f"[Federated Learning] Total aggregation rounds: {fed_stats['total_aggregations']}")

        # Print coordinator statistics
        coordinator_stats = self.coordinator.get_stats()
        print(f"[Coordinator] RB conflicts: {coordinator_stats['total_rb_conflicts']}, "
              f"Power conflicts: {coordinator_stats['total_power_conflicts']}")

        return {
            'rewards': self.episode_rewards,
            'delays': self.episode_delays,
            'energies': self.episode_energies,
            'constraint_satisfaction': self.episode_constraint_satisfaction,
        }

    def evaluate(self, num_episodes: int = None) -> Dict:
        """
        Evaluate trained HMDQN model
        Args:
            num_episodes: number of evaluation episodes

        Returns:
            dict with evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.config.EVAL_EPISODES

        print(f"\n[Evaluation] Evaluating HMDQN with {num_episodes} episodes...")

        all_delays = []
        all_energies = []
        all_rewards = []
        constraint_violations = 0

        for episode in range(num_episodes):
            obs_list = self.env.reset()
            episode_reward = 0

            for step in range(self.config.LOCAL_STEPS):
                # Greedy action selection (no exploration)
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.choose_action(obs_list[i], exploration=False)
                    actions.append(action)

                actions = np.array(actions)
                next_obs_list, rewards, done, info = self.env.step(actions)

                episode_reward += np.sum(rewards)
                all_delays.extend(info['delays'])
                all_energies.extend(info['energies'])

                if info['constraint_violated']:
                    constraint_violations += 1

                obs_list = next_obs_list

            all_rewards.append(episode_reward)

        results = {
            'avg_delay': float(np.mean(all_delays)) if len(all_delays) > 0 else 0.0,
            'std_delay': float(np.std(all_delays)) if len(all_delays) > 0 else 0.0,
            'avg_energy': float(np.mean(all_energies)) if len(all_energies) > 0 else 0.0,
            'std_energy': float(np.std(all_energies)) if len(all_energies) > 0 else 0.0,
            'avg_reward': float(np.mean(all_rewards)) if len(all_rewards) > 0 else 0.0,
            'constraint_satisfaction_rate': 1.0 - (constraint_violations / max(len(all_delays), 1)),
            'delays_array': np.array(all_delays),
            'energies_array': np.array(all_energies),
        }

        return results


class Evaluator:
    """Evaluator for comparing different allocation methods"""

    def __init__(self, cfg):
        self.config = cfg

    def evaluate_all_methods(self, trainer: FMDQNTrainer, num_episodes: int = 10) -> Dict:
        """
        Evaluate all allocation methods
        Args:
            trainer: trained HMDQN trainer
            num_episodes: evaluation episodes per method

        Returns:
            dict with results for all methods
        """
        results = {}

        # 1. FMDQN (trained agents with federated learning)
        print("\n[Evaluate] Testing FMDQN...")
        results['FMDQN'] = trainer.evaluate(num_episodes)

        # 2. Random allocation
        print("[Evaluate] Testing Random allocation...")
        results['Random'] = self._eval_baseline(trainer.env, RandomAllocation(
            self.config.N_VEHICLES, self.config), num_episodes)

        # 3. Equal allocation
        print("[Evaluate] Testing Equal allocation...")
        results['Equal'] = self._eval_baseline(trainer.env, EqualAllocation(
            self.config.N_VEHICLES, self.config), num_episodes)

        # 4. Greedy allocation
        print("[Evaluate] Testing Greedy allocation...")
        results['Greedy'] = self._eval_baseline(trainer.env, GreedyAllocation(
            self.config.N_VEHICLES, self.config), num_episodes)

        return results

    def _eval_baseline(self, env: IoVEnv, allocator, num_episodes: int) -> Dict:
        """
        Evaluate a baseline allocation method
        Args:
            env: environment
            allocator: allocation strategy object
            num_episodes: number of episodes

        Returns:
            dict with evaluation metrics
        """
        all_delays = []
        all_energies = []
        constraint_violations = 0

        for episode in range(num_episodes):
            obs_list = env.reset()

            for step in range(self.config.LOCAL_STEPS):
                # Baseline allocation logic
                channel_gains = env.channel.get_channel_gain()
                queue_sizes = np.array([v.queue_size for v in env.vehicles])

                # Allocate continuous actions then discretize
                continuous_actions = allocator.allocate(channel_gains, queue_sizes)

                # Convert continuous [0,1] to discrete actions
                discrete_actions = []
                for i in range(len(continuous_actions)):
                    bw_ratio = continuous_actions[i, 0]
                    power_ratio = continuous_actions[i, 1]
                    cpu_ratio = continuous_actions[i, 2]

                    rb_idx = int(np.round(bw_ratio * (self.config.N_RB_ACTIONS - 1)))
                    power_idx = int(np.round(power_ratio * (self.config.N_POWER_ACTIONS - 1)))
                    cpu_idx = int(np.round(cpu_ratio * (self.config.N_CPU_ACTIONS - 1)))

                    discrete_action = (rb_idx * (self.config.N_POWER_ACTIONS * self.config.N_CPU_ACTIONS) +
                                     power_idx * self.config.N_CPU_ACTIONS + cpu_idx)
                    discrete_actions.append(discrete_action)

                discrete_actions = np.array(discrete_actions)
                next_obs_list, _, done, info = env.step(discrete_actions)

                all_delays.extend(info['delays'])
                all_energies.extend(info['energies'])

                if info['constraint_violated']:
                    constraint_violations += 1

                obs_list = next_obs_list

        results = {
            'avg_delay': np.mean(all_delays),
            'std_delay': np.std(all_delays),
            'avg_energy': np.mean(all_energies),
            'std_energy': np.std(all_energies),
            'constraint_satisfaction_rate': 1.0 - (constraint_violations / (len(all_delays) + 1e-10)),
            'delays_array': np.array(all_delays),
            'energies_array': np.array(all_energies),
        }

        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FMDQN-based IoV Resource Allocation')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval'],
                       help='Operating mode: train or eval')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (default: config.FL_ROUNDS)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create directories
    create_directories(config)

    if args.mode == 'train':
        # Training mode
        trainer = FMDQNTrainer(config)
        train_results = trainer.train(args.episodes)

        # Save results
        timestamp = get_time_str()
        results_path = f"{config.RESULTS_DIR}/train_results_{timestamp}.json"

        results_dict = {
            'config': config.get_config_dict(),
            'train_rewards': train_results['rewards'],
            'train_delays': train_results['delays'],
            'train_energies': train_results['energies'],
            'constraint_satisfaction': train_results['constraint_satisfaction'],
        }

        save_results(results_dict, results_path)

        # Plot convergence
        fig, ax = plot_convergence(train_results['rewards'],
                                  title="FMDQN Training Convergence (Federated Learning)")
        plt.savefig(f"{config.RESULTS_DIR}/convergence_{timestamp}.png", dpi=150)
        plt.close()

        print(f"\n[Main] Training results saved to {results_path}")

    elif args.mode == 'eval':
        # Evaluation mode
        print("[Main] Training FMDQN for evaluation (20 episodes)...")

        trainer = FMDQNTrainer(config)
        trainer.train(num_episodes=20)

        # Evaluation
        evaluator = Evaluator(config)
        eval_results = evaluator.evaluate_all_methods(trainer, num_episodes=5)

        # Save results
        timestamp = get_time_str()
        results_path = f"{config.RESULTS_DIR}/eval_results_{timestamp}.json"

        eval_results_dict = {}
        for method, results in eval_results.items():
            eval_results_dict[method] = {
                'avg_delay': float(results['avg_delay']),
                'std_delay': float(results['std_delay']),
                'avg_energy': float(results['avg_energy']),
                'std_energy': float(results['std_energy']),
                'constraint_satisfaction_rate': float(results['constraint_satisfaction_rate']),
            }

        save_results(eval_results_dict, results_path)

        # Print results
        print_results(eval_results_dict, title="FMDQN vs Baseline Comparison")

        # Plotting
        fig, ax = plot_comparison(
            {method: {'avg_delay': r['avg_delay'],
                     'avg_energy': r['avg_energy'],
                     'constraint_satisfaction_rate': r['constraint_satisfaction_rate']}
             for method, r in eval_results.items()}
        )
        plt.savefig(f"{config.RESULTS_DIR}/comparison_{timestamp}.png", dpi=150)
        plt.close()

        # Delay distribution
        fig, ax = plot_delay_distribution(
            [eval_results[m]['delays_array'] for m in eval_results.keys()],
            list(eval_results.keys())
        )
        plt.savefig(f"{config.RESULTS_DIR}/delays_{timestamp}.png", dpi=150)
        plt.close()

        print(f"\n[Main] Evaluation results saved to {results_path}")


if __name__ == '__main__':
    main()
