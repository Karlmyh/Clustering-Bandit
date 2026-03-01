import numpy as np
from typing import Dict, Iterator, Optional, Tuple

from ..config import DOMAIN_LOW, DOMAIN_HIGH
from .node import TreeNode
from .splitter import MaxEdgeRandomSplitter


class BinaryTreePartition:
    """Binary tree partition of [0,1]^d with nodes storing samples.
    
    Each leaf node stores its samples (xs, ys) directly. When a leaf
    splits, samples are redistributed to children.
    """
    
    def __init__(
        self,
        d: int,
        rng: Optional[np.random.Generator] = None,
        splitter: Optional[object] = None,
    ):
        """Initialize the partition with a single root node covering [0,1]^d.
        
        Args:
            d: Dimension of the context space.
            rng: Random number generator for tie-breaking.
            splitter: Optional splitter object for determining splits.
        """
        self.d = d
        self.rng = rng if rng is not None else np.random.default_rng()
        self.splitter = splitter if splitter is not None else MaxEdgeRandomSplitter(rng=self.rng)
        
        # Dictionary mapping node_id -> TreeNode
        self.nodes: Dict[int, TreeNode] = {}
        
        # Set of leaf node IDs
        self.leaf_ids: set[int] = set()
        
        # Next available node ID
        self.next_node_id = 0
        
        # Initialize with root node
        root = TreeNode(
            node_id=self.next_node_id,
            low=np.full(d, DOMAIN_LOW),
            high=np.full(d, DOMAIN_HIGH),
            depth=0,
            parent_id=None,
        )
        self.nodes[root.node_id] = root
        self.leaf_ids.add(root.node_id)
        self.next_node_id += 1
    
    def find_leaf(self, x: np.ndarray) -> int:
        """Find the leaf node ID containing point x.
        
        Args:
            x: Point in [0,1]^d (shape: (d,)).
            
        Returns:
            ID of the leaf node containing x.
        """
        x = np.asarray(x)
        assert x.shape == (self.d,), f"x must have shape ({self.d},), got {x.shape}"
        # Validate domain bounds
        assert np.all(x >= DOMAIN_LOW), f"x must be >= {DOMAIN_LOW} (domain is [0,1]^d)"
        assert np.all(x <= DOMAIN_HIGH), f"x must be <= {DOMAIN_HIGH} (domain is [0,1]^d)"
        
        # Start from root and traverse to leaf
        current_id = 0  # Root is always node_id 0
        
        while True:
            node = self.nodes[current_id]
            
            # If this is a leaf, return it
            if node.is_leaf():
                return current_id
            
            # Otherwise, route to appropriate child
            current_id = node.route(x)

    def find_leafs(self, X: np.ndarray) -> np.ndarray:
        """Find the leaf node ID containing point x.
        
        Args:
            X: Points in [0,1]^d (shape: (n,d)).
            
        Returns:
            Arrary of IDs of the leaf node containing X.
        """
        X = np.asarray(X)
        assert X.shape[1] == self.d, f"X must have shape (n,{self.d}), got {X.shape}"
        # domain
        assert np.all(X >= DOMAIN_LOW), f"X must be >= {DOMAIN_LOW} (domain is [0,1]^d)"
        assert np.all(X <= DOMAIN_HIGH), f"X must be <= {DOMAIN_HIGH} (domain is [0,1]^d)"
        
        result_nodeid = np.zeros(X.shape[0], dtype=np.int32)
        # Start from root and traverse to leaf
        for i in range(X.shape[0]):
            node_id = 0
            while True:
                node = self.nodes[node_id]
                if node.is_leaf():
                    break
                node_id = node.route(X[i])
            result_nodeid[i] = node_id
        return result_nodeid
    

    
    def get_node(self, node_id: int) -> TreeNode:
        """Get the TreeNode object for a given node_id.
        
        Args:
            node_id: ID of the node.
            
        Returns:
            The TreeNode object.
        """
        return self.nodes[node_id]
    
    def add_sample(self, x: np.ndarray, y: float) -> int:

        """Add a sample to the appropriate leaf node.
        
        Args:
            x: Context (shape: (d,)).
            y: Reward.
            
        Returns:
            ID of the leaf node that received the sample.
        """
        leaf_id = self.find_leaf(x)
        self.nodes[leaf_id].add_sample(x, y)
        return leaf_id

    def add_samples(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Add samples to the appropriate leaf nodes.
        
        Args:
            X: Points in [0,1]^d (shape: (n,d)).
            y: Rewards (shape: (n,)).
            
        Returns:
            Arrary of IDs of the leaf node containing X.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        result_nodeid = self.find_leafs(X)
        for i in range(X.shape[0]):
            self.nodes[result_nodeid[i]].add_sample(X[i], y[i])
        return result_nodeid
    
    def split_leaf(self, leaf_id: int) -> Tuple[int, int]:
        """Split a leaf node along its longest edge.
        
        Redistributes all samples in the node to its children.
        
        Args:
            leaf_id: ID of the leaf node to split.
            
        Returns:
            Tuple of (left_child_id, right_child_id).
        """
        if leaf_id not in self.leaf_ids:
            raise ValueError(f"Node {leaf_id} is not a leaf and cannot be split")
        
        leaf_node = self.nodes[leaf_id]
        
        # Create new child nodes
        left_id = self.next_node_id
        right_id = self.next_node_id + 1
        self.next_node_id += 2
        
        # Split the node using splitter (this redistributes samples)
        left_node, right_node = leaf_node.split(left_id, right_id, splitter=self.splitter)
        
        # Add children to partition
        self.nodes[left_id] = left_node
        self.nodes[right_id] = right_node
        
        # Remove parent from leaves, add children
        self.leaf_ids.remove(leaf_id)
        self.leaf_ids.add(left_id)
        self.leaf_ids.add(right_id)
        
        return left_id, right_id
    
    def iter_leaves(self) -> Iterator[int]:
        """Iterate over the leaf nodes."""
        return iter(self.leaf_ids)
    
    
    def num_leaves(self) -> int:
        """Get the number of leaf nodes."""
        return len(self.leaf_ids)
    
    def get_node_range(self, node_id: int) -> np.ndarray:
        """Get the boundaries of a node.
        
        Args:
            node_id: ID of the node.
            
        Returns:
            Array of shape (2, d) - [low, high].
        """
        node = self.nodes[node_id]
        return np.array([node.low, node.high])
    