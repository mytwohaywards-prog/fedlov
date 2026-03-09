"""
Utility functions for training, evaluation, and visualization
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from datetime import datetime


def create_directories(config):
    """Create necessary directories for saving results and models"""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    return {
        'results_dir': config.RESULTS_DIR,
        'models_dir': config.MODELS_DIR,
    }


def save_results(data: Dict, filepath: str):
    """
    Save results dictionary to JSON file
    Args:
        data: dictionary with results
        filepath: path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_checkpoint(agents: List, server, filepath: str):
    """Save agents and server checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'agent_buffers': [agent.buffer.size for agent in agents],
        'fl_round': server.round_count,
    }

    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)

    # Also save server state
    server_path = filepath.replace('.pkl', '_server.pt')
    server.save_checkpoint(server_path)

    print(f"Checkpoint saved to {filepath}")


def plot_convergence(episode_rewards: List[float], title: str = "Training Convergence",
                    smoothing_window: int = 10):
    """
    Plot reward convergence curve during training
    Args:
        episode_rewards: list of rewards per episode/round
        title: plot title
        smoothing_window: window for moving average smoothing
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw rewards
    ax.plot(episode_rewards, alpha=0.3, label='Episode Reward')

    # Plot smoothed rewards (✅ 改: 修复绘图错误)
    if smoothing_window > 1 and len(episode_rewards) >= smoothing_window:
        smoothed = np.convolve(episode_rewards, np.ones(smoothing_window) / smoothing_window,
                              mode='valid')
        if len(smoothed) > 0:  # 确保smoothed不为空
            x_smooth = np.arange(smoothing_window - 1, len(episode_rewards))
            ax.plot(x_smooth[:len(smoothed)], smoothed,
                   linewidth=2, label=f'Smoothed (window={smoothing_window})')

    ax.set_xlabel('Episode / FL Round')
    ax.set_ylabel('Average Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_comparison(results_dict: Dict[str, Dict],
                   metrics: List[str] = ['avg_delay', 'avg_energy', 'constraint_satisfaction_rate'],
                   figsize: Tuple = (15, 5)):
    """
    Plot comparison of different allocation methods
    Args:
        results_dict: dict with keys as method names, values as result dicts
        metrics: list of metrics to plot
        figsize: figure size

    Returns:
        fig, axes for further customization
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    methods = list(results_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        values = [results_dict[method].get(metric, 0) for method in methods]

        bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, axes


def plot_delay_distribution(delays_list: List[np.ndarray], method_names: List[str]):
    """
    Plot distribution of delays for different methods
    Args:
        delays_list: list of delay arrays from different methods
        method_names: names of methods
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(delays_list, labels=method_names)
    ax.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='T_MAX constraint')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Latency Distribution by Allocation Method')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    return fig, ax


def plot_energy_distribution(energies_list: List[np.ndarray], method_names: List[str]):
    """
    Plot distribution of energy consumption for different methods
    Args:
        energies_list: list of energy arrays from different methods
        method_names: names of methods
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(energies_list, labels=method_names)
    ax.set_ylabel('Energy (Joules)')
    ax.set_title('Energy Consumption Distribution by Allocation Method')
    ax.grid(True, alpha=0.3, axis='y')

    return fig, ax


def plot_scalability(vehicle_counts: List[int], performance_results: Dict[str, List[float]],
                    metric: str = 'avg_delay'):
    """
    Plot scalability: performance vs number of vehicles
    Args:
        vehicle_counts: list of vehicle counts
        performance_results: dict with method names as keys, list of metrics as values
        metric: metric name for y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, values in performance_results.items():
        ax.plot(vehicle_counts, values, marker='o', linewidth=2, label=method_name)

    ax.set_xlabel('Number of Vehicles')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Scalability: {metric.replace("_", " ").title()} vs Vehicle Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def compute_statistics(results_dict: Dict) -> Dict:
    """
    Compute summary statistics from results
    Args:
        results_dict: dict with metric arrays

    Returns:
        statistics: dict with mean, std, min, max for each metric
    """
    stats = {}

    for metric_name, values in results_dict.items():
        if isinstance(values, (list, np.ndarray)):
            values = np.array(values)
            stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }

    return stats


def print_results(results_dict: Dict, title: str = "Results"):
    """Pretty print results"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

    for method_name, metrics in results_dict.items():
        print(f"\n{method_name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric:.<40} {value:.6f}")
            else:
                print(f"  {metric:.<40} {value}")

    print("=" * 60 + "\n")


def get_time_str() -> str:
    """Get current time as formatted string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
