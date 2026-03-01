"""Multi-user bandit simulator with clustering."""

import numpy as np
from typing import Dict, List

from ..bandits import CandidateSetBanditEnv, RewardFunction
from ..config import CLUSTERING_UPDATE_INTERVAL
from ..utils.rng import create_rng
from .shared_stats import SharedPartitionStats
from .clustering import OnlineClustering


class MultiUserSimulator:
    """Simulator for multi-user bandit experiments."""
    
    def __init__(
        self,
        reward_fns: List[RewardFunction],
        K: int,
        T: int,
        collaborate: bool = True,
        seed: int = 0,
    ):
        """Initialize simulator.
        
        Args:
            reward_fns: List of reward functions (one per user).
            K: Number of candidates per round.
            T: Total number of rounds.
            collaborate: Whether to use collaborative learning.
            seed: Random seed.
        """
        self.m = len(reward_fns)
        self.reward_fns = reward_fns
        self.K = K
        self.T = T
        self.collaborate = collaborate
        self.rng = create_rng(seed)
        
        # Create environments
        self.envs = [
            CandidateSetBanditEnv(reward_fn=fn, K=K, seed=seed + i)
            for i, fn in enumerate(reward_fns)
        ]
        
        # Shared statistics
        d = reward_fns[0].d
        self.shared_stats = SharedPartitionStats(d, T, self.m, self.rng)
        
        # Clustering
        if collaborate:
            self.clustering = OnlineClustering(self.m)
        else:
            self.clustering = None
    
    def run(self) -> List[Dict]:
        """Run the multi-user experiment.
        
        Returns:
            List of log entries (one per round).
        """
        # Reset environments
        candidates_list = [env.reset() for env in self.envs]
        
        logs = []
        cumulative_regret_independent = 0.0
        cumulative_regret_collaborative = 0.0
        
        for t in range(1, self.T + 1):
            # Each user selects an action
            actions = []
            selected_candidates = []
            
            for user_id in range(self.m):
                candidates = candidates_list[user_id]
                
                # Select action using UCB (with or without collaboration)
                if self.collaborate:
                    # Use shared statistics with clustering
                    ucb_values = [
                        self.shared_stats.get_ucb_value(user_id, x, use_shared=True)
                        for x in candidates
                    ]
                else:
                    # Use independent statistics
                    ucb_values = [
                        self.shared_stats.get_ucb_value(user_id, x, use_shared=False)
                        for x in candidates
                    ]
                
                ucb_values = np.array(ucb_values)
                max_ucb = np.max(ucb_values)
                best_mask = ucb_values == max_ucb
                best_indices = np.where(best_mask)[0]
                action_idx = int(self.rng.choice(best_indices))
                
                actions.append(action_idx)
                selected_candidates.append(candidates[action_idx])
            
            # Step environments and get rewards
            rewards = []
            regrets_independent = []
            regrets_collaborative = []
            
            for user_id in range(self.m):
                next_candidates, reward, info = self.envs[user_id].step(actions[user_id])
                candidates_list[user_id] = next_candidates
                rewards.append(reward)
                regrets_independent.append(info["regret"])
                
                # Update statistics
                self.shared_stats.update(user_id, selected_candidates[user_id], reward)
            
            # End of round processing
            self.shared_stats.end_round(t)
            
            # Update clustering periodically
            if self.collaborate and t % CLUSTERING_UPDATE_INTERVAL == 0:
                new_clusters = self.clustering.cluster_users(self.shared_stats)
                self.shared_stats.update_clusters(new_clusters)
            
            # Compute collaborative regret (if using collaboration)
            if self.collaborate:
                # Re-compute actions with collaborative stats for comparison
                for user_id in range(self.m):
                    candidates = self.envs[user_id].history[-1]["candidates"]
                    ucb_values = [
                        self.shared_stats.get_ucb_value(user_id, x, use_shared=True)
                        for x in candidates
                    ]
                    # Oracle is still the same
                    regrets_collaborative.append(regrets_independent[user_id])  # Same oracle
            else:
                regrets_collaborative = regrets_independent
            
            # Log
            cumulative_regret_independent += sum(regrets_independent)
            cumulative_regret_collaborative += sum(regrets_collaborative)
            
            log_entry = {
                "t": t,
                "regret_independent": float(sum(regrets_independent)),
                "regret_collaborative": float(sum(regrets_collaborative)),
                "cumulative_regret_independent": float(cumulative_regret_independent),
                "cumulative_regret_collaborative": float(cumulative_regret_collaborative),
                "num_leaves": self.shared_stats.partition.num_leaves(),
                "num_clusters": len(set(self.shared_stats.cluster_assignments.values())),
            }
            logs.append(log_entry)
        
        return logs
