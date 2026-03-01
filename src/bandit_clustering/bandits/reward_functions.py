"""Reward functions for synthetic bandit environments."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    def __init__(self, d: int, rng: Optional[np.random.Generator] = None):
        """Initialize reward function.
        
        Args:
            d: Dimension of the context space.
            rng: Random number generator for initialization.
        """
        self.d = d
        self.rng = rng if rng is not None else np.random.default_rng()
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate reward at point x.
        
        Args:
            x: Point in [0,1]^d (shape: (d,)).
            
        Returns:
            Reward value in [0,1].
        """
        pass
    
    @abstractmethod
    def get_global_optimum(self) -> Tuple[float, np.ndarray]:
        """Get the global optimum (exact, not estimated).
        
        Returns:
            Tuple of (max_reward, argmax_point).
        """
        pass


class LinearReward(RewardFunction):
    """Linear reward function: f(x) = x[0] 
    
    The global optimum is at x[0] = 1
    """
    
    def __init__(
        self,
        d: int,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize linear reward.
        
        Args:
            d: Dimension of the context space.
            weights: Weight vector (shape: (d,)). If None, randomly sampled from [-1, 1].
            bias: Bias term (default: 0.0).
            rng: Random number generator.
        """
        super().__init__(d, rng)
        

        # Precompute the global optimum
        self._optimum_point = np.ones(d)
        
        
        self._optimum_value = 1
    
    def __call__(self, X: np.ndarray) -> float:
        """Evaluate linear reward.
        
        Args:
            x: Point in [0,1]^d (shape: (n, d)).

        """
        
        return X.mean(axis=1)
    
    def get_global_optimum(self) -> Tuple[float, np.ndarray]:
        """Get the global optimum (exact, predefined).
        
        Returns:
            Tuple of (max_reward, argmax_point).
        """
        return self._optimum_value, self._optimum_point.copy()

    def get_oracle_reward(self, candidates: np.ndarray) -> float:
        """Get the reward of the best candidate.
        
        Args:
            candidates: Array of candidates (shape: (K, d)).
            
        Returns:
            Reward of the best candidate.
        """
        return np.max(self.__call__(candidates))