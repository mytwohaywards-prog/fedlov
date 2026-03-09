from .dqn_network import (
    QNetwork, DuelingQNetwork,
    soft_update_dqn, hard_update_dqn,
    get_q_network_parameters, set_q_network_parameters,
    initialize_weights
)

# Legacy imports (kept for backward compatibility)
from .networks import (
    ActorNetwork, CriticNetwork, DualCriticNetwork, ActorCriticNetwork,
    soft_update, hard_update,
    get_network_parameters, set_network_parameters
)

__all__ = [
    # DQN networks (primary for HMDQN)
    'QNetwork', 'DuelingQNetwork',
    'soft_update_dqn', 'hard_update_dqn',
    'get_q_network_parameters', 'set_q_network_parameters',
    'initialize_weights',
    # Legacy (kept for reference)
    'ActorNetwork', 'CriticNetwork', 'DualCriticNetwork', 'ActorCriticNetwork',
    'soft_update', 'hard_update',
    'get_network_parameters', 'set_network_parameters'
]
