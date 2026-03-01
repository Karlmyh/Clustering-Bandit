"""Baseline agents for comparison."""

import numpy as np
from typing import Optional

from ..utils.rng import create_rng


class RandomAgent:
    """Random selection baseline."""
    
    def __init__(self, K: int, seed: Optional[int] = None):
        """Initialize random agent.
        
        Args:
            K: Number of candidates per round.
            seed: Random seed.
        """
        self.K = K
        self.rng = create_rng(seed)
    
    def select_action(self, candidates: np.ndarray, t: int) -> int:
        """Select a random candidate.
        
        Args:
            candidates: Array of candidates (unused).
            t: Current round (unused).
            
        Returns:
            Random index.
        """
        return int(self.rng.integers(0, self.K))
    
    def update(self, x: np.ndarray, y: float) -> None:
        """Update (no-op for random agent)."""
        pass
    
    def end_round(self, t: int) -> None:
        """End round (no-op)."""
        pass


class EpsilonGreedyAgent:
    """Epsilon-greedy agent with histogram estimates.
    
    Uses the same partition structure as AdaptivePartitionUCB but
    selects greedily most of the time.
    """
    
    def __init__(
        self,
        d: int,
        T: int,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize epsilon-greedy agent.
        
        Args:
            d: Dimension.
            T: Total rounds.
            epsilon: Exploration probability.
            seed: Random seed.
        """
        self.d = d
        self.T = T
        self.epsilon = epsilon
        self.rng = create_rng(seed)
        
        # Use same partition structure as AdaptivePartitionUCB
        from ..partition import BinaryTreePartition
        self.partition = BinaryTreePartition(d, self.rng)
    
    def select_action(self, candidates: np.ndarray, t: int) -> int:
        """Select action using epsilon-greedy.
        
        Args:
            candidates: Array of candidates.
            t: Current round.
            
        Returns:
            Selected index.
        """
        if self.rng.random() < self.epsilon:
            # Explore: random
            return int(self.rng.integers(0, len(candidates)))
        else:
            # Exploit: greedy based on empirical means
            K = len(candidates)
            mean_values = []
            
            for x in candidates:
                leaf_id = self.partition.find_leaf(x)
                node = self.partition.get_node(leaf_id)
                mean = node.get_mean()
                mean_values.append(mean)
            
            mean_values = np.array(mean_values)
            max_mean = np.max(mean_values)
            best_mask = mean_values == max_mean
            best_indices = np.where(best_mask)[0]
            return int(self.rng.choice(best_indices))
    
    def update(self, x: np.ndarray, y: float) -> None:
        """Update statistics."""
        self.partition.add_sample(x, y)
    
    def end_round(self, t: int) -> None:
        """End round processing."""
        # Use same splitting logic as AdaptivePartitionUCB
        from ..config import C_SPLIT
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            splits_performed = False
            
            leaves_to_check = list(self.partition.iter_leaves())
            
            for leaf_id in leaves_to_check:
                if leaf_id not in self.partition.leaf_ids:
                    continue
                
                node = self.partition.get_node(leaf_id)
                count = node.get_count()
                diameter = node.diameter
                
                if diameter > 0 and count > 0:
                    threshold = C_SPLIT * np.log(self.T) / (diameter ** 2)
                    if count >= threshold:
                        self.partition.split_leaf(leaf_id)
                        splits_performed = True
            
            if not splits_performed:
                break
