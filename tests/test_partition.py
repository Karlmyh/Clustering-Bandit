"""Tests for partition structure (TreeNode and BinaryTreePartition)."""

import numpy as np
import pytest

from src.bandit_clustering.partition import TreeNode, BinaryTreePartition
from src.bandit_clustering.config import DOMAIN_LOW, DOMAIN_HIGH
from src.bandit_clustering.utils.rng import create_rng


class TestTreeNode:
    """Tests for TreeNode class."""
    
    def test_node_initialization(self):
        """Test node initialization."""
        d = 2
        node = TreeNode(
            node_id=0,
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            depth=0,
        )
        
        assert node.node_id == 0
        assert node.depth == 0
        assert node.is_leaf()
        assert node.get_count() == 0
        assert node.get_mean() == 0.0
    
    def test_node_properties(self):
        """Test node geometric properties."""
        node = TreeNode(
            node_id=0,
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 0.5]),
            depth=0,
        )
        
        assert node.diameter == 1.0  # max(1.0, 0.5)
        assert node.volume == 0.5  # 1.0 * 0.5
    
    def test_node_contains(self):
        """Test point containment check."""
        node = TreeNode(
            node_id=0,
            low=np.array([0.2, 0.3]),
            high=np.array([0.8, 0.7]),
            depth=0,
        )
        
        assert node.contains(np.array([0.5, 0.5]))
        assert node.contains(np.array([0.2, 0.3]))  # Boundary
        assert node.contains(np.array([0.8, 0.7]))  # Boundary
        assert not node.contains(np.array([0.1, 0.5]))
        assert not node.contains(np.array([0.9, 0.5]))
    
    def test_node_add_sample(self):
        """Test adding samples to a node."""
        node = TreeNode(
            node_id=0,
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            depth=0,
        )
        
        node.add_sample(np.array([0.5, 0.5]), 0.7)
        assert node.get_count() == 1
        assert node.get_mean() == 0.7
        
        node.add_sample(np.array([0.6, 0.6]), 0.8)
        assert node.get_count() == 2
        assert abs(node.get_mean() - 0.75) < 1e-10
    
    def test_node_split_max_edge(self):
        """Test splitting a node along max edge."""
        node = TreeNode(
            node_id=0,
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 0.5]),  # Dimension 0 is longest
            depth=0,
        )
        
        # Add some samples
        node.add_sample(np.array([0.2, 0.2]), 0.5)
        node.add_sample(np.array([0.8, 0.3]), 0.9)
        
        # Split
        left_node, right_node = node.split(left_id=1, right_id=2)
        
        # Check parent is no longer a leaf
        assert not node.is_leaf()
        assert node.feature == 0  # Split on dimension 0
        assert node.threshold == 0.5  # Midpoint
        
        # Check children
        assert left_node.node_id == 1
        assert right_node.node_id == 2
        assert left_node.is_leaf()
        assert right_node.is_leaf()
        
        # Check samples redistributed
        assert left_node.get_count() == 1  # x[0] = 0.2 < 0.5
        assert right_node.get_count() == 1  # x[0] = 0.8 >= 0.5
        
        # Parent should have no samples
        assert node.get_count() == 0
    
    def test_node_split_with_splitter(self):
        """Test splitting with a splitter."""
        from src.bandit_clustering.partition.splitter import GainReductionSplitter
        
        node = TreeNode(
            node_id=0,
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            depth=0,
        )
        
        # Add samples with clear separation
        node.add_sample(np.array([0.2, 0.5]), 0.1)
        node.add_sample(np.array([0.3, 0.5]), 0.1)
        node.add_sample(np.array([0.8, 0.5]), 0.9)
        node.add_sample(np.array([0.9, 0.5]), 0.9)
        
        splitter = GainReductionSplitter(search_number=5)
        left_node, right_node = node.split(left_id=1, right_id=2, splitter=splitter)
        
        # Should split on dimension 0 (where separation is clear)
        assert node.feature == 0
        assert left_node.get_count() == 2
        assert right_node.get_count() == 2
    
    def test_node_route(self):
        """Test routing a point to a child."""
        node = TreeNode(
            node_id=0,
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            depth=0,
        )
        
        # Split the node
        left_node, right_node = node.split(left_id=1, right_id=2)
        
        # Route points
        assert node.route(np.array([0.3, 0.5])) == 1  # Left child
        assert node.route(np.array([0.7, 0.5])) == 2  # Right child
        assert node.route(np.array([0.5, 0.5])) == 2  # At threshold, goes right


class TestBinaryTreePartition:
    """Tests for BinaryTreePartition class."""
    
    def test_partition_initialization(self):
        """Test partition initialization."""
        d = 2
        partition = BinaryTreePartition(d, rng=create_rng(0))
        
        assert partition.d == d
        assert partition.num_leaves() == 1
        assert 0 in partition.leaf_ids
        
        root = partition.get_node(0)
        assert root.is_leaf()
        assert np.all(root.low == DOMAIN_LOW)
        assert np.all(root.high == DOMAIN_HIGH)
    
    def test_partition_find_leaf(self):
        """Test finding leaf containing a point."""
        d = 2
        partition = BinaryTreePartition(d, rng=create_rng(0))
        
        leaf_id = partition.find_leaf(np.array([0.5, 0.5]))
        assert leaf_id == 0  # Root
        
        # After splitting, should find correct leaf
        partition.split_leaf(0)
        leaf_id_left = partition.find_leaf(np.array([0.2, 0.5]))
        leaf_id_right = partition.find_leaf(np.array([0.8, 0.5]))
        assert leaf_id_left != leaf_id_right
        assert leaf_id_left in partition.leaf_ids
        assert leaf_id_right in partition.leaf_ids
    
    def test_partition_add_sample(self):
        """Test adding samples to partition."""
        d = 2
        partition = BinaryTreePartition(d, rng=create_rng(0))
        
        leaf_id = partition.add_sample(np.array([0.5, 0.5]), 0.7)
        assert leaf_id == 0
        
        node = partition.get_node(0)
        assert node.get_count() == 1
        assert node.get_mean() == 0.7
    
    def test_partition_split_leaf(self):
        """Test splitting a leaf node."""
        d = 2
        partition = BinaryTreePartition(d, rng=create_rng(0))
        
        # Add samples
        partition.add_sample(np.array([0.2, 0.5]), 0.5)
        partition.add_sample(np.array([0.8, 0.5]), 0.9)
        
        # Split
        left_id, right_id = partition.split_leaf(0)
        
        assert partition.num_leaves() == 2
        assert 0 not in partition.leaf_ids
        assert left_id in partition.leaf_ids
        assert right_id in partition.leaf_ids
        
        # Check samples redistributed
        left_node = partition.get_node(left_id)
        right_node = partition.get_node(right_id)
        assert left_node.get_count() == 1
        assert right_node.get_count() == 1
    
    def test_partition_apply(self):
        """Test apply method (routing multiple points)."""
        d = 2
        partition = BinaryTreePartition(d, rng=create_rng(0))
        
        # Split once
        partition.split_leaf(0)
        
        # Route multiple points
        X = np.array([
            [0.2, 0.5],
            [0.8, 0.5],
            [0.5, 0.5],
        ])
        leaf_ids = partition.find_leafs(X)
        
        assert len(leaf_ids) == 3
        assert all(lid in partition.leaf_ids for lid in leaf_ids)
    
    def test_partition_data_retention(self):
        """Test that all samples are retained after splits."""
        d = 2
        partition = BinaryTreePartition(d, rng=create_rng(0))
        
        # Add many samples
        samples = []
        for i in range(10):
            x = np.array([0.1 + i * 0.08, 0.5])
            y = 0.5 + i * 0.05
            partition.add_sample(x, y)
            samples.append((x, y))
        
        # Split multiple times
        for _ in range(3):
            leaves = list(partition.iter_leaves())
            if len(leaves) > 0:
                partition.split_leaf(leaves[0])
        
        # Count total samples in all leaves
        total_count = sum(partition.get_node(lid).get_count() for lid in partition.iter_leaves())
        assert total_count == 10  # All samples retained
    
    def test_partition_iter_leaves(self):
        """Test iterating over leaves."""
        d = 2
        partition = BinaryTreePartition(d, rng=create_rng(0))
        
        leaves = list(partition.iter_leaves())
        assert len(leaves) == 1
        
        partition.split_leaf(0)
        leaves = list(partition.iter_leaves())
        assert len(leaves) == 2
