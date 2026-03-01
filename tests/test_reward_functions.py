"""Tests for reward functions."""

import numpy as np

from src.bandit_clustering.bandits.reward_functions import LinearReward, QuadraticReward
from src.bandit_clustering.utils.rng import create_rng


def test_linear_reward_initialization():
    """Test LinearReward initialization."""
    d = 2
    reward_fn = LinearReward(d, rng=create_rng(0))
    
    assert reward_fn.d == d
    assert reward_fn._optimum_point.shape == (d,)
    assert np.all(reward_fn._optimum_point == 1.0)  # Optimum at [1, 1, ...]


def test_linear_reward_evaluation():
    """Test LinearReward evaluation."""
    d = 2
    reward_fn = LinearReward(d, rng=create_rng(0))
    
    # Test at different points
    x1 = np.array([0.0, 0.0])
    x2 = np.array([0.5, 0.5])
    x3 = np.array([1.0, 1.0])
    
    r1 = reward_fn(x1)
    r2 = reward_fn(x2)
    r3 = reward_fn(x3)
    
    # Reward should increase with mean of x
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
    assert 0.0 <= r3 <= 1.0
    assert r1 < r2 < r3  # Since f(x) = mean(x)


def test_linear_reward_optimum():
    """Test LinearReward global optimum."""
    d = 3
    reward_fn = LinearReward(d, rng=create_rng(0))
    
    opt_value, opt_point = reward_fn.get_global_optimum()
    
    # Optimum should be at [1, 1, 1, ...]
    assert np.all(opt_point == 1.0)
    
    # Optimum value should be 1.0 (mean of all 1s)
    assert abs(opt_value - 1.0) < 1e-10
    
    # Verify it's actually the maximum
    test_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([1.0, 0.0, 0.0]),
        opt_point,
    ]
    rewards = [reward_fn(x) for x in test_points]
    assert max(rewards) == opt_value


def test_linear_reward_determinism():
    """Test that reward function is deterministic with same seed."""
    d = 2
    reward_fn1 = LinearReward(d, rng=create_rng(42))
    reward_fn2 = LinearReward(d, rng=create_rng(42))
    
    x = np.array([0.5, 0.5])
    assert reward_fn1(x) == reward_fn2(x)
    
    opt1 = reward_fn1.get_global_optimum()
    opt2 = reward_fn2.get_global_optimum()
    assert opt1[0] == opt2[0]
    assert (opt1[1] == opt2[1]).all()


def test_linear_reward_domain():
    """Test that reward function works across domain."""
    d = 2
    reward_fn = LinearReward(d, rng=create_rng(0))
    
    # Test points at boundaries and interior
    test_points = [
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
    ]
    
    for x in test_points:
        reward = reward_fn(x)
        assert 0.0 <= reward <= 1.0


def test_quadratic_reward_initialization():
    """Test QuadraticReward initialization."""
    d = 3
    reward_fn = QuadraticReward(d, rng=create_rng(0))

    assert reward_fn.d == d
    assert reward_fn._optimum_point.shape == (d,)
    assert np.all(reward_fn._optimum_point == 1.0)


def test_quadratic_reward_evaluation_and_optimum():
    """Quadratic reward should peak at x=[1,...,1] and stay in [0, 1]."""
    d = 2
    reward_fn = QuadraticReward(d, rng=create_rng(0))

    x1 = np.array([0.0, 0.0])
    x2 = np.array([0.5, 0.5])
    x3 = np.array([1.0, 1.0])

    r1 = reward_fn(x1)
    r2 = reward_fn(x2)
    r3 = reward_fn(x3)

    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
    assert 0.0 <= r3 <= 1.0
    assert r1 < r2 < r3

    opt_value, opt_point = reward_fn.get_global_optimum()
    assert np.all(opt_point == 1.0)
    assert abs(opt_value - 1.0) < 1e-10


def test_reward_function_vectorized_input():
    """Reward functions should support vectorized input (n, d)."""
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    linear = LinearReward(2, rng=create_rng(0))
    quadratic = QuadraticReward(2, rng=create_rng(0))

    linear_values = linear(X)
    quadratic_values = quadratic(X)

    assert linear_values.shape == (3,)
    assert quadratic_values.shape == (3,)
