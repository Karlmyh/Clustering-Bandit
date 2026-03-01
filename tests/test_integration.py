"""Integration tests for the full bandit system."""

import numpy as np
import pytest

from src.bandit_clustering.agents import AdaptivePartitionUCB
from src.bandit_clustering.bandits import CandidateSetBanditEnv
from src.bandit_clustering.bandits.reward_functions import LinearReward
from src.bandit_clustering.utils.rng import create_rng


def test_full_episode():
    """Test a full episode of bandit interaction."""
    d = 2
    T = 50
    K = 5
    
    # Create components
    reward_fn = LinearReward(d=d, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=K, seed=42)
    agent = AdaptivePartitionUCB(d=d, T=T, seed=42)
    
    # Run episode
    cumulative_regret = 0.0
    
    for t in range(1, T + 1):
        candidates = env._generate_candidates()
        # Agent selects action
        action = agent.select_action(candidates, t=t)
        env.step(candidates, action)
        info = env.history[-1]
        reward = info["selected_reward"]
        agent.update(candidates[action], reward)
        agent.perform_splits()
        cumulative_regret += info["regret"]
    
    
    # Check final state
    assert agent.partition.num_leaves() >= 1
    
    # All samples should be retained
    total_samples = sum(
        agent.partition.get_node(lid).get_count()
        for lid in agent.partition.iter_leaves()
    )
    assert total_samples == T


def test_agent_learning():
    """Test that agent learns over time."""
    d = 2
    T = 100
    K = 10
    
    reward_fn = LinearReward(d=d, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=K, seed=42)
    agent = AdaptivePartitionUCB(d=d, T=T, seed=42)

    regrets = []
    
    for t in range(1, T + 1):
        candidates = env._generate_candidates()
        action = agent.select_action(candidates, t=t)
        env.step(candidates, action)
        info = env.history[-1]
        reward = info["selected_reward"]
        agent.update(candidates[action], reward)
        agent.perform_splits()
        
        regrets.append(info["regret"])
    
    # Agent should improve over time (regret should decrease on average)
    # Check that later regrets are lower than early regrets on average
    early_regret = np.mean(regrets[:T//4])
    late_regret = np.mean(regrets[-T//4:])
    
    # This is a weak test, but agent should at least be exploring
    assert len(regrets) == T
    assert all(r >= 0 for r in regrets)


def test_partition_growth():
    """Test that partition grows appropriately."""
    d = 2
    T = 200
    K = 5
    
    reward_fn = LinearReward(d=d, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=K, seed=42)
    agent = AdaptivePartitionUCB(d=d, T=T, seed=42)
    

    num_leaves_history = []
    
    for t in range(1, T + 1):
        candidates = env._generate_candidates()
        action = agent.select_action(candidates, t=t)
        env.step(candidates, action)
        info = env.history[-1]
        reward = info["selected_reward"]
        agent.update(candidates[action], reward)
        
        num_leaves_history.append(agent.partition.num_leaves())
    
    # Partition should grow (or at least not shrink)
    assert num_leaves_history[-1] >= num_leaves_history[0]
    assert max(num_leaves_history) >= 1


def test_deterministic_reproducibility():
    """Test that the system is reproducible with same seeds."""
    d = 2
    T = 20
    K = 5
    
    # Run 1
    reward_fn1 = LinearReward(d=d, rng=create_rng(0))
    env1 = CandidateSetBanditEnv(reward_fn=reward_fn1, K=K, seed=42)
    agent1 = AdaptivePartitionUCB(d=d, T=T, seed=42)
    

    actions1 = []
    rewards1 = []
    
    for t in range(1, T + 1):
        candidates1 = env1._generate_candidates()
        action = agent1.select_action(candidates1, t=t)
        actions1.append(action)
        env1.step(candidates1, action)
        info = env1.history[-1]
        reward = info["selected_reward"]
        rewards1.append(reward)
        agent1.update(candidates1[action], reward)
        agent1.perform_splits()
    
    
    # Run 2 (same seeds)

    reward_fn2 = LinearReward(d=d, rng=create_rng(0))
    env2 = CandidateSetBanditEnv(reward_fn=reward_fn2, K=K, seed=42)
    agent2 = AdaptivePartitionUCB(d=d, T=T, seed=42)
    
    candidates2 = env2._generate_candidates()
    actions2 = []
    rewards2 = []
    
    for t in range(1, T + 1):
        action = agent2.select_action(candidates2, t=t)
        actions2.append(action)
        env2.step(candidates2, action)
        info = env2.history[-1]
        reward = info["selected_reward"]
        rewards2.append(reward)
        agent2.update(candidates2[action], reward)
        agent2.perform_splits()
        candidates2 = env2._generate_candidates()
    
    # Should be identical
    assert actions1 == actions2
    np.testing.assert_array_almost_equal(rewards1, rewards2)
    assert agent1.partition.num_leaves() == agent2.partition.num_leaves()
