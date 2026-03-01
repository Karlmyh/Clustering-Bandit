"""Adaptive partition UCB agent using nodes that store samples directly."""

import numpy as np
from typing import Optional
from collections import deque

from ..config import C_SPLIT, UCB_CONFIDENCE_MULTIPLIER, APPROXIMATION_ERROR_FACTOR
from ..partition import BinaryTreePartition
from ..utils.rng import create_rng
from ..partition.splitter import MaxEdgeRandomSplitter, GainReductionMaxEdgeSplitter, GainReductionSplitter


class AdaptivePartitionUCB:
    """Adaptive partition UCB agent for candidate-set bandits.
    
    Maintains an adaptive partition where leaf nodes store samples directly.
    Uses node-wise UCB to select actions from candidate sets.
    """
    
    def __init__(
        self,
        d: int,
        T: int,
        seed: Optional[int] = None,
    ):
        """Initialize the agent.
        
        Args:
            d: Dimension of the context space.
            T: Total number of rounds (for UCB confidence and splitting).
            seed: Random seed.
        """
        self.d = d
        self.T = T
        self.rng = create_rng(seed)
        
        # Partition structure (nodes store samples directly)
        self.partition = BinaryTreePartition(d, self.rng)

    def _get_ucb_width(self, node_id: int) -> float:
        """Get UCB width for a node.
        
        Args:
            node_id: ID of the leaf node.
            
        Returns:
            UCB width.
        """
        node = self.partition.get_node(node_id)
        count = node.get_count()
        return np.sqrt(UCB_CONFIDENCE_MULTIPLIER * np.log(self.T) / count)
    
    def _get_ucb_value(self, node_id: int) -> float:
        """Get UCB value for a node.
        
        Args:
            node_id: ID of the leaf node.
            
        Returns:
            UCB value = mean + width + approx (or +inf if no samples).
        """
        node = self.partition.get_node(node_id)
        count = node.get_count()
        
        if count == 0:
            return np.inf
        
        # edge_id, edge_length = node.get_longest_edge()
        # approximation_err = edge_length * APPROXIMATION_ERROR_FACTOR
        mean = node.get_mean()
        width = np.sqrt(UCB_CONFIDENCE_MULTIPLIER * np.log(self.T) / count) 
        return mean + width 

    def get_ucb_widths(self, X: np.ndarray) -> np.ndarray:
        """Get UCB widths for a set of points.
        
        Args:
            X: Points in [0,1]^d (shape: (n,d)).
            
        Returns:
            UCB widths (shape: (n,)).
        """
        X = np.asarray(X)
        ucb_widths = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            leaf_id = self.partition.find_leaf(X[i])
            ucb_widths[i] = self._get_ucb_width(leaf_id)
        return ucb_widths
    
    def get_ucbs(self, X: np.ndarray) -> np.ndarray:
        """Get UCB values for a set of points.
        
        Args:
            X: Points in [0,1]^d (shape: (n,d)).
            
        Returns:
            UCB values (shape: (n,)).
        """
        X = np.asarray(X)
        ucb_values = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            leaf_id = self.partition.find_leaf(X[i])
            ucb_values[i] = self._get_ucb_value(leaf_id)
        return ucb_values
    
    def select_action(self, candidates: np.ndarray, t: int) -> int:
        """Select an action from the candidate set using UCB.
        
        Args:
            candidates: Array of candidate contexts (shape: (K, d)).
            t: Current round (1-indexed, for logging).
            
        Returns:
            Index of the selected candidate.
        """
        K = len(candidates)
        
        ucb_values = self.get_ucbs(candidates)
        max_ucb = np.max(ucb_values)
        
        # Find all candidates with maximum UCB
        best_mask = ucb_values == max_ucb
        
        if np.sum(best_mask) == 1:
            return int(np.argmax(best_mask))
        
        # Tie-breaking: uniform random among best candidates
        best_indices = np.where(best_mask)[0]
        return int(self.rng.choice(best_indices))
    
    def update(self, x: np.ndarray, y: float) -> None:
        """Update statistics with a new observation.
        
        The sample is added directly to the appropriate leaf node.
        
        Args:
            x: Selected context.
            y: Observed reward.
        """
        # Add sample to the appropriate leaf node
        # (samples are stored directly in nodes)
        self.partition.add_sample(x, y)

    def perform_splits(self) -> None:
        """Perform splits on the partition."""
        leaf_ids = list(self.partition.iter_leaves())
        for leaf_id in leaf_ids:
            node = self.partition.get_node(leaf_id)
            count = node.get_count()
            diameter = node.diameter
            if diameter > 0 and count > 0:
                threshold = C_SPLIT * np.log(self.T) / (diameter ** 2)
                if count >= threshold:
                    self.partition.split_leaf(leaf_id)
                   
        return self

