"""Reward functions for synthetic bandit environments."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


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

    def _coerce_input(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Normalize input to 2D array and track whether caller passed one sample."""
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            if X.shape[0] != self.d:
                raise ValueError(f"Expected shape ({self.d},), got {X.shape}")
            return X.reshape(1, -1), True
        if X.ndim == 2:
            if X.shape[1] != self.d:
                raise ValueError(f"Expected shape (n, {self.d}), got {X.shape}")
            return X, X.shape[0] == 1
        raise ValueError(f"Expected 1D or 2D input, got ndim={X.ndim}")


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
    
    def __call__(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate linear reward.
        
        Args:
            x: Point in [0,1]^d (shape: (n, d)).

        """
        
        X2d, single = self._coerce_input(X)
        values = X2d.mean(axis=1)
        if single:
            return float(values[0])
        return values
    
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
        return float(np.max(self.__call__(candidates)))


class QuadraticReward(RewardFunction):
    """Quadratic reward: f(x) = 1 - mean((1 - x)^2).

    The global optimum is at x = [1, ..., 1] with value 1.
    """

    def __init__(
        self,
        d: int,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(d, rng)
        self._optimum_point = np.ones(d)
        self._optimum_value = 1.0

    def __call__(self, X: np.ndarray) -> Union[float, np.ndarray]:
        X2d, single = self._coerce_input(X)
        values = 1.0 - np.mean((1.0 - X2d) ** 2, axis=1)
        if single:
            return float(values[0])
        return values

    def get_global_optimum(self) -> Tuple[float, np.ndarray]:
        return self._optimum_value, self._optimum_point.copy()

    def get_oracle_reward(self, candidates: np.ndarray) -> float:
        return float(np.max(self.__call__(candidates)))
