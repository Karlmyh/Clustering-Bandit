"""Online clustering for multi-user bandits."""

import numpy as np
from typing import Dict, List, Set

from ..config import CLUSTERING_DISAGREEMENT_THRESHOLD
from .shared_stats import SharedPartitionStats


class OnlineClustering:
    """Online clustering based on empirical disagreement in high-value regions."""
    
    def __init__(
        self,
        m: int,
        threshold: float = CLUSTERING_DISAGREEMENT_THRESHOLD,
    ):
        """Initialize clustering.
        
        Args:
            m: Number of users.
            threshold: Disagreement threshold for clustering.
        """
        self.m = m
        self.threshold = threshold
    
    def compute_disagreement(
        self,
        user_i: int,
        user_j: int,
        shared_stats: SharedPartitionStats,
        top_k_leaves: int = 10,
    ) -> float:
        """Compute empirical disagreement between two users.
        
        Args:
            user_i: ID of first user.
            user_j: ID of second user.
            shared_stats: Shared partition statistics.
            top_k_leaves: Number of top-value leaves to consider.
            
        Returns:
            Disagreement score.
        """
        # Get leaves with high visit counts or high estimated values
        leaves = list(shared_stats.partition.iter_leaves())
        
        # Score leaves by visit count and estimated value
        leaf_scores = []
        for leaf_id in leaves:
            count_i = shared_stats.user_stats[user_i].get_count(leaf_id)
            count_j = shared_stats.user_stats[user_j].get_count(leaf_id)
            mean_i = shared_stats.user_stats[user_i].get_mean(leaf_id)
            mean_j = shared_stats.user_stats[user_j].get_mean(leaf_id)
            
            # Score by total visits and value
            score = count_i + count_j + abs(mean_i) + abs(mean_j)
            leaf_scores.append((leaf_id, score))
        
        # Sort by score and take top k
        leaf_scores.sort(key=lambda x: x[1], reverse=True)
        top_leaves = [leaf_id for leaf_id, _ in leaf_scores[:top_k_leaves]]
        
        # Compute weighted disagreement
        total_weight = 0.0
        total_disagreement = 0.0
        
        for leaf_id in top_leaves:
            count_i = shared_stats.user_stats[user_i].get_count(leaf_id)
            count_j = shared_stats.user_stats[user_j].get_count(leaf_id)
            mean_i = shared_stats.user_stats[user_i].get_mean(leaf_id)
            mean_j = shared_stats.user_stats[user_j].get_mean(leaf_id)
            
            if count_i > 0 or count_j > 0:
                weight = count_i + count_j
                disagreement = abs(mean_i - mean_j)
                total_weight += weight
                total_disagreement += weight * disagreement
        
        if total_weight == 0:
            return 1.0  # High disagreement if no shared data
        
        return total_disagreement / total_weight
    
    def cluster_users(self, shared_stats: SharedPartitionStats) -> Dict[int, int]:
        """Cluster users based on disagreement.
        
        Args:
            shared_stats: Shared partition statistics.
            
        Returns:
            Dictionary mapping user_id -> cluster_id.
        """
        # Compute pairwise disagreements
        disagreements = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(i + 1, self.m):
                d = self.compute_disagreement(i, j, shared_stats)
                disagreements[i, j] = d
                disagreements[j, i] = d
        
        # Build adjacency graph: users are connected if disagreement < threshold
        adjacency = disagreements < self.threshold
        
        # Find connected components (clusters)
        visited = [False] * self.m
        cluster_assignments: Dict[int, int] = {}
        cluster_id = 0
        
        def dfs(node: int, cluster: int):
            """Depth-first search to find connected component."""
            visited[node] = True
            cluster_assignments[node] = cluster
            for neighbor in range(self.m):
                if adjacency[node, neighbor] and not visited[neighbor]:
                    dfs(neighbor, cluster)
        
        for i in range(self.m):
            if not visited[i]:
                dfs(i, cluster_id)
                cluster_id += 1
        
        return cluster_assignments
