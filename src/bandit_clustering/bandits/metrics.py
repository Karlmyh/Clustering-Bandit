"""Metrics for evaluating bandit performance."""

import numpy as np
from typing import Callable


def compute_regret(
    candidates: np.ndarray,
    selected_idx: int,
    reward_fn: Callable[[np.ndarray], float],
) -> float:
    """Compute candidate-oracle regret for a single round.
    
    Args:
        candidates: Array of candidate contexts (shape: (K, d)).
        selected_idx: Index of the selected candidate.
        reward_fn: Reward function f: [0,1]^d -> [0,1].
        
    Returns:
        Regret = f(x★) - f(x_selected), where x★ is the best candidate.
    """
    rewards = np.array([reward_fn(x) for x in candidates])
    best_reward = np.max(rewards)
    selected_reward = rewards[selected_idx]
    return float(best_reward - selected_reward)
