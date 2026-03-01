"""Shared partition statistics for multi-user collaboration.

Uses a shared BinaryTreePartition where splitting is driven by pooled counts,
but each user maintains their own samples per leaf node.
"""

import numpy as np
from typing import Dict, List, Optional

from ..config import C_SPLIT, UCB_CONFIDENCE_MULTIPLIER
from ..partition import BinaryTreePartition


class SharedPartitionStats:
    """Maintains shared partition and per-user statistics.
    
    All users share the same partition structure, but each user
    maintains their own samples per leaf node. Statistics can be pooled
    within clusters for collaboration.
    """
    
    def __init__(self, d: int, T: int, m: int, rng: np.random.Generator):
        """Initialize shared partition stats.
        
        Args:
            d: Dimension.
            T: Total rounds.
            m: Number of users.
            rng: Random number generator.
        """
        self.d = d
        self.T = T
        self.m = m
        self.rng = rng
        
        # Shared partition (nodes don't store samples directly for multi-user)
        # Instead, we'll track samples per user per leaf
        self.partition = BinaryTreePartition(d, rng)
        
        # Per-user samples per leaf: user_id -> (leaf_id -> list of (x, y))
        self.user_samples: List[Dict[int, List[tuple]]] = [{} for _ in range(m)]
        
        # Cluster assignments: user_id -> cluster_id
        self.cluster_assignments: Dict[int, int] = {i: i for i in range(m)}
    
    def update(self, user_id: int, x: np.ndarray, y: float) -> None:
        """Update statistics for a user.
        
        Args:
            user_id: ID of the user.
            x: Selected context.
            y: Observed reward.
        """
        leaf_id = self.partition.find_leaf(x)
        
        # Store sample for this user in this leaf
        if leaf_id not in self.user_samples[user_id]:
            self.user_samples[user_id][leaf_id] = []
        self.user_samples[user_id][leaf_id].append((x.copy(), y))
    
    def _get_user_count(self, user_id: int, leaf_id: int) -> int:
        """Get sample count for a user in a leaf.
        
        Args:
            user_id: ID of the user.
            leaf_id: ID of the leaf node.
            
        Returns:
            Sample count.
        """
        return len(self.user_samples[user_id].get(leaf_id, []))
    
    def _get_user_mean(self, user_id: int, leaf_id: int) -> float:
        """Get mean reward for a user in a leaf.
        
        Args:
            user_id: ID of the user.
            leaf_id: ID of the leaf node.
            
        Returns:
            Mean reward (0.0 if no samples).
        """
        samples = self.user_samples[user_id].get(leaf_id, [])
        if len(samples) == 0:
            return 0.0
        return float(np.mean([y for _, y in samples]))
    
    def _get_pooled_count(self, leaf_id: int, cluster_users: Optional[List[int]] = None) -> int:
        """Get pooled sample count for a leaf.
        
        Args:
            leaf_id: ID of the leaf node.
            cluster_users: List of user IDs to pool (None = all users).
            
        Returns:
            Pooled count.
        """
        if cluster_users is None:
            cluster_users = list(range(self.m))
        
        total = 0
        for user_id in cluster_users:
            total += self._get_user_count(user_id, leaf_id)
        return total
    
    def _get_pooled_mean(self, leaf_id: int, cluster_users: Optional[List[int]] = None) -> float:
        """Get pooled mean reward for a leaf.
        
        Args:
            leaf_id: ID of the leaf node.
            cluster_users: List of user IDs to pool (None = all users).
            
        Returns:
            Pooled mean (0.0 if no samples).
        """
        if cluster_users is None:
            cluster_users = list(range(self.m))
        
        total_count = 0
        total_sum = 0.0
        
        for user_id in cluster_users:
            samples = self.user_samples[user_id].get(leaf_id, [])
            for _, y in samples:
                total_count += 1
                total_sum += y
        
        if total_count == 0:
            return 0.0
        return total_sum / total_count
    
    def get_ucb_value(self, user_id: int, x: np.ndarray, use_shared: bool = False) -> float:
        """Get UCB value for a user at point x.
        
        Args:
            user_id: ID of the user.
            x: Context point.
            use_shared: If True, use pooled statistics from user's cluster.
            
        Returns:
            UCB value.
        """
        leaf_id = self.partition.find_leaf(x)
        
        if use_shared:
            # Pool statistics from all users in the same cluster
            cluster_id = self.cluster_assignments[user_id]
            cluster_users = [uid for uid, cid in self.cluster_assignments.items() if cid == cluster_id]
            
            count = self._get_pooled_count(leaf_id, cluster_users)
            if count == 0:
                return np.inf
            
            mean = self._get_pooled_mean(leaf_id, cluster_users)
            width = np.sqrt(UCB_CONFIDENCE_MULTIPLIER * np.log(self.T) / count)
            return mean + width
        else:
            # Use user's own statistics
            count = self._get_user_count(user_id, leaf_id)
            if count == 0:
                return np.inf
            
            mean = self._get_user_mean(user_id, leaf_id)
            width = np.sqrt(UCB_CONFIDENCE_MULTIPLIER * np.log(self.T) / count)
            return mean + width
    
    def end_round(self, t: int) -> None:
        """End of round: check and perform splits.
        
        Splitting is driven by pooled counts across all users.
        When a leaf splits, all users' samples are redistributed.
        
        Args:
            t: Current round.
        """
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            splits_performed = False
            
            leaves_to_check = list(self.partition.iter_leaves())
            
            for leaf_id in leaves_to_check:
                if leaf_id not in self.partition.leaf_ids:
                    continue
                
                # Get pooled count across all users
                total_count = self._get_pooled_count(leaf_id)
                
                if total_count == 0:
                    continue
                
                node = self.partition.get_node(leaf_id)
                diameter = node.diameter
                
                if diameter > 0:
                    threshold = C_SPLIT * np.log(self.T) / (diameter ** 2)
                    if total_count >= threshold:
                        # Split the node
                        left_id, right_id = self.partition.split_leaf(leaf_id)
                        
                        # Redistribute all users' samples
                        self._redistribute_samples(leaf_id, left_id, right_id)
                        
                        splits_performed = True
            
            if not splits_performed:
                break
    
    def _redistribute_samples(self, old_leaf_id: int, left_id: int, right_id: int) -> None:
        """Redistribute samples from a split leaf to its children.
        
        Args:
            old_leaf_id: ID of the leaf that was split.
            left_id: ID of left child.
            right_id: ID of right child.
        """
        left_node = self.partition.get_node(left_id)
        right_node = self.partition.get_node(right_id)
        
        # Redistribute samples for each user
        for user_id in range(self.m):
            samples = self.user_samples[user_id].get(old_leaf_id, [])
            
            if len(samples) == 0:
                continue
            
            left_samples = []
            right_samples = []
            
            for x, y in samples:
                if left_node.contains(x):
                    left_samples.append((x, y))
                elif right_node.contains(x):
                    right_samples.append((x, y))
                else:
                    # On boundary: assign randomly
                    if self.rng.random() < 0.5:
                        left_samples.append((x, y))
                    else:
                        right_samples.append((x, y))
            
            # Update user's samples
            if len(left_samples) > 0:
                self.user_samples[user_id][left_id] = left_samples
            if len(right_samples) > 0:
                self.user_samples[user_id][right_id] = right_samples
            
            # Remove old leaf samples
            if old_leaf_id in self.user_samples[user_id]:
                del self.user_samples[user_id][old_leaf_id]
    
    def update_clusters(self, new_assignments: Dict[int, int]) -> None:
        """Update cluster assignments.
        
        Args:
            new_assignments: Dictionary mapping user_id -> cluster_id.
        """
        self.cluster_assignments = new_assignments.copy()
