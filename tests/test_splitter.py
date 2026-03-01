"""Tests for splitter classes."""

import numpy as np
import pytest

from src.bandit_clustering.partition.splitter import GainReductionSplitter, GainReductionMaxEdgeSplitter
from src.bandit_clustering.utils.rng import create_rng


def test_gain_reduction_splitter_basic():
    """Test GainReductionSplitter with basic data."""
    splitter = GainReductionSplitter(search_number=5, rng=create_rng(0))
    
    X = np.array([
        [0.2, 0.5],
        [0.3, 0.5],
        [0.8, 0.5],
        [0.9, 0.5],
    ])
    y = np.array([0.1, 0.1, 0.9, 0.9])
    X_range = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    split_dim, split_point = splitter(X, X_range, y)
    
    assert len(split_dim) > 0
    assert len(split_point) > 0
    assert len(split_dim) == len(split_point)
    
    # Should prefer dimension 0 (where separation is clear)
    assert split_dim[0] == 0
    assert 0.0 < split_point[0] < 1.0


def test_gain_reduction_splitter_empty_data():
    """Test GainReductionSplitter with empty data."""
    splitter = GainReductionSplitter(rng=create_rng(0))
    
    X = np.empty((0, 2))
    y = np.empty(0)
    X_range = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    split_dim, split_point = splitter(X, X_range, y)
    
    # Should handle empty data gracefully
    assert isinstance(split_dim, list)
    assert isinstance(split_point, list)
    assert len(split_dim) == 1  # One per time
    assert len(split_point) == 1


def test_gain_reduction_max_edge_splitter():
    """Test GainReductionMaxEdgeSplitter."""
    splitter = GainReductionMaxEdgeSplitter(rng=create_rng(0))
    
    X = np.array([
        [0.2, 0.5],
        [0.3, 0.5],
        [0.8, 0.5],
        [0.9, 0.5],
    ])
    y = np.array([0.1, 0.1, 0.9, 0.9])
    X_range = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    split_dim, split_point = splitter(X, X_range, y)
    
    assert len(split_dim) > 0
    assert len(split_point) > 0
    
    # Should only consider max-edge dimensions
    edge_lengths = X_range[1] - X_range[0]
    max_length = np.max(edge_lengths)
    for dim in split_dim:
        assert edge_lengths[dim] == max_length


def test_gain_reduction_max_edge_splitter_rectangular():
    """Test GainReductionMaxEdgeSplitter with rectangular region."""
    splitter = GainReductionMaxEdgeSplitter(rng=create_rng(0))
    
    X = np.array([
        [0.2, 0.5],
        [0.3, 0.5],
        [0.8, 0.5],
        [0.9, 0.5],
    ])
    y = np.array([0.1, 0.1, 0.9, 0.9])
    X_range = np.array([[0.0, 0.0], [1.0, 0.5]])  # Dimension 0 is longest
    
    split_dim, split_point = splitter(X, X_range, y)
    
    # Should only split on dimension 0
    assert all(dim == 0 for dim in split_dim)


def test_splitter_determinism():
    """Test that splitters are deterministic with same seed."""
    X = np.array([
        [0.2, 0.5],
        [0.3, 0.5],
        [0.8, 0.5],
        [0.9, 0.5],
    ])
    y = np.array([0.1, 0.1, 0.9, 0.9])
    X_range = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    splitter1 = GainReductionSplitter(search_number=5, rng=create_rng(42))
    splitter2 = GainReductionSplitter(search_number=5, rng=create_rng(42))
    
    dim1, point1 = splitter1(X, X_range, y)
    dim2, point2 = splitter2(X, X_range, y)
    
    assert dim1 == dim2
    assert point1 == point2
