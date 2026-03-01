"""Tests for CandidateSetBanditEnv."""

import numpy as np
import pytest

from src.bandit_clustering.bandits import CandidateSetBanditEnv
from src.bandit_clustering.bandits.reward_functions import LinearReward
from src.bandit_clustering.utils.rng import create_rng


def test_env_initialization():
    """Test environment initialization."""
    reward_fn = LinearReward(d=2, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=10, seed=42)
    
    assert env.K == 10
    assert env.d == 2
    assert len(env.history) == 0


def test_env_generate_candidates():
    """Test candidate generation."""
    reward_fn = LinearReward(d=2, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=5, seed=42)
    
    candidates = env._generate_candidates()
    
    assert candidates.shape == (5, 2)
    assert np.all((candidates >= 0) & (candidates <= 1))



def test_env_step():
    """Test environment step."""
    reward_fn = LinearReward(d=2, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=5, seed=42)
    
    # Generate candidates first
    candidates = env._generate_candidates()
    
    # Take a step
    action_idx = 0
    env.step(candidates, action_idx)

    info = env.history[-1]
    reward = info["selected_reward"]
    
    assert 0.0 <= reward <= 1.0
    assert "oracle_reward" in info
    assert "regret" in info
    assert "optimal_reward" in info
    
    # Check regret is non-negative
    assert info["regret"] >= 0
    
    # Check oracle reward is at least as good as selected
    assert info["oracle_reward"] >= reward


def test_env_oracle_selection():
    """Test that oracle selects best candidate."""
    reward_fn = LinearReward(d=2, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=5, seed=42)
    
    candidates = env._generate_candidates()
    
    # Compute rewards manually
    rewards = np.array([reward_fn(x) for x in candidates])
    oracle_idx_manual = int(np.argmax(rewards))
    
    # Take step with some action
    env.step(candidates, action_idx=0)
    info = env.history[-1]
    
    # Oracle should match manua
    # l computation
    assert info["optimal_reward"] == oracle_idx_manual
    assert abs(info["oracle_reward"] - rewards[oracle_idx_manual]) < 1e-2


def test_env_multiple_steps():
    """Test environment over multiple steps."""
    reward_fn = LinearReward(d=2, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=5, seed=42)
    
    candidates = env._generate_candidates()
    
    cumulative_regret = 0.0
    for t in range(5):
        action = t % 5  # Cycle through actions
        env.step(candidates, action)
        info = env.history[-1]
        reward = info["selected_reward"]
        cumulative_regret += info["regret"]
        # Generate new candidates for next round
        candidates = env._generate_candidates()
    

    assert len(env.history) == 5  # 1 initial + 5 steps
    assert cumulative_regret >= 0


def test_env_determinism():
    """Test that environment is deterministic with same seed."""
    reward_fn = LinearReward(d=2, rng=create_rng(0))
    
    env1 = CandidateSetBanditEnv(reward_fn=reward_fn, K=5, seed=42)
    env2 = CandidateSetBanditEnv(reward_fn=reward_fn, K=5, seed=42)
    
    candidates1 = env1._generate_candidates()
    candidates2 = env2._generate_candidates()
    
    np.testing.assert_array_equal(candidates1, candidates2)
    
    # Same action should give same reward
    env1.step(candidates1, 0)
    env2.step(candidates2, 0)
    reward1 = env1.history[-1]["selected_reward"]
    reward2 = env2.history[-1]["selected_reward"]
    
    assert abs(reward1 - reward2) < 1e-10


def test_env_candidate_generation():
    """Test candidate generation."""
    reward_fn = LinearReward(d=3, rng=create_rng(0))
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=10, seed=42)
    
    candidates = env._generate_candidates()
    
    # Check shape and domain
    assert candidates.shape == (10, 3)
    assert np.all((candidates >= 0) & (candidates <= 1))
    
    # Check candidates are different
    assert len(np.unique(candidates, axis=0)) > 1  # At least some are different
