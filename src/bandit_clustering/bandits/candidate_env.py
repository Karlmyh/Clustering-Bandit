"""Candidate-set bandit environment."""

import numpy as np
from typing import Dict, Optional, Tuple

from ..utils.rng import create_rng
from .reward_functions import RewardFunction


class CandidateSetBanditEnv:
    """Environment for candidate-set contextual bandits.
    
    At each round, generates K candidate contexts from P_X and provides
    rewards when a candidate is selected.
    """
    
    def __init__(
        self,
        reward_fn: RewardFunction,
        K: int,
        candidate_distribution: str = "uniform",
        seed: Optional[int] = None,
    ):
        """Initialize the environment.
        
        Args:
            reward_fn: Reward function f: [0,1]^d -> [0,1].
            K: Number of candidates per round.
            candidate_distribution: Distribution for candidates ("uniform").
            seed: Random seed.
        """
        self.reward_fn = reward_fn
        self.K = K
        self.d = reward_fn.d
        self.candidate_distribution = candidate_distribution
        self.rng = create_rng(seed)
        

        self.history: list[Dict] = []
    
   
    
    def step(self, candidates: np.ndarray, action_idx: int) -> "CandidateSetBanditEnv":
        """Step the environment.
        
        Args:
            candidates: Array of candidates (shape: (K, d)).
            action_idx: Index of the selected candidate.
            
        Returns:
            self
        """
        
        optimal_reward = self.reward_fn.get_global_optimum()[0]
        oracle_reward = self.reward_fn.get_oracle_reward(candidates)
        selected_reward = self.reward_fn(candidates[action_idx].reshape(1, -1))
        regret = oracle_reward - selected_reward
        global_regret = optimal_reward - selected_reward
        noisy_reward = selected_reward + self.rng.normal(0, 0.1)
        selected_candidate = candidates[action_idx]
        self.history.append({
            "optimal_reward": optimal_reward, 
            "oracle_reward": oracle_reward, 
            "selected_reward": selected_reward, 
            "regret": regret, 
            "global_regret": global_regret,
            "candidates": candidates.copy(),
            "selected_candidate": selected_candidate,
            "noisy_reward": noisy_reward,
        })
        
        
        return selected_candidate, noisy_reward
    
    def _generate_candidates(self) -> np.ndarray:
        """Generate K candidate contexts from P_X.
        
        Generates candidates uniformly from [0,1]^d (the default domain).
        
        Returns:
            Array of candidates (shape: (K, d)), each in [0,1]^d.
        """
        if self.candidate_distribution == "uniform":
            # Generate uniform random samples in [0,1]^d
            candidates = self.rng.random((self.K, self.d))
        else:
            raise ValueError(f"Unknown candidate distribution: {self.candidate_distribution}")
        
        
        return candidates
