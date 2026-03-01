"""Tests for criterion functions."""

import numpy as np
import pytest

from src.bandit_clustering.partition.criterion import insample_ssq, mse


def test_insample_ssq():
    """Test in-sample sum of squares."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = insample_ssq(y)
    
    # var = 1.25, so ssq = 1.25 * 4 = 5.0
    expected = np.var(y) * len(y)
    assert abs(result - expected) < 1e-10
    
    # Empty array
    assert insample_ssq(np.array([])) == 0.0


def test_mse_basic():
    """Test MSE criterion with basic data."""
    X = np.array([
        [0.2, 0.5],
        [0.3, 0.5],
        [0.8, 0.5],
        [0.9, 0.5],
    ])
    Y = np.array([0.1, 0.1, 0.9, 0.9])
    X_range = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    # Split on dimension 0 at 0.5
    result = mse(X, X_range, Y, d=0, split=0.5)
    
    # Left: [0.1, 0.1], Right: [0.9, 0.9]
    # Left variance is 0, right variance is 0
    # So MSE reduction should be high (negative of sum of variances)
    assert result > -np.inf
    assert isinstance(result, float)


def test_mse_empty_data():
    """Test MSE with empty data."""
    X = np.empty((0, 2))
    Y = np.empty(0)
    X_range = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    result = mse(X, X_range, Y, d=0, split=0.5)
    assert result == -np.inf


def test_mse_perfect_separation():
    """Test MSE with perfect separation."""
    X = np.array([
        [0.1, 0.5],
        [0.2, 0.5],
        [0.8, 0.5],
        [0.9, 0.5],
    ])
    Y = np.array([0.0, 0.0, 1.0, 1.0])
    X_range = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    # Split at 0.5 should give perfect separation
    result = mse(X, X_range, Y, d=0, split=0.5)
    
    # Both sides have zero variance, so result should be 0
    assert result == 0.0


def test_mse_no_separation():
    """Test MSE when split doesn't help."""
    X = np.array([
        [0.2, 0.5],
        [0.3, 0.5],
        [0.4, 0.5],
        [0.5, 0.5],
    ])
    Y = np.array([0.5, 0.5, 0.5, 0.5])  # All same value
    X_range = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    result = mse(X, X_range, Y, d=0, split=0.35)
    
    # Variance is 0 everywhere, so result should be 0
    assert result == 0.0
