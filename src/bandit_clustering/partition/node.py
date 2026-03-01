import numpy as np
from typing import Optional, Tuple

from ..config import DOMAIN_LOW, DOMAIN_HIGH, UCB_CONFIDENCE_MULTIPLIER

_TREE_LEAF = -1


class TreeNode:
    """A node in the binary tree partition.
    
    Leaf nodes store samples (x, y) directly. When a leaf splits,
    its samples are redistributed to its two children.
    
    Attributes:
        node_id: Unique identifier for this node.
        low: Lower bounds for each dimension (shape: (d,)).
        high: Upper bounds for each dimension (shape: (d,)).
        depth: Depth in the tree (0 for root).
        parent_id: ID of parent node (None for root).
        left_child_id: ID of left child (None if leaf).
        right_child_id: ID of right child (None if leaf).
        feature: Splitting dimension (None for leaves).
        threshold: Splitting point (None for leaves).
        
        # Sample storage (only for leaves)
        xs: Array of sample contexts (shape: (n, d)).
        ys: Array of sample rewards (shape: (n,)).
    """
    
    def __init__(
        self,
        node_id: int,
        low: np.ndarray,
        high: np.ndarray,
        depth: int = 0,
        parent_id: Optional[int] = None,
    ):
        """Initialize a tree node.
        
        Args:
            node_id: Unique identifier.
            low: Lower bounds (shape: (d,)).
            high: Upper bounds (shape: (d,)).
            depth: Depth in tree.
            parent_id: Parent node ID.
        """
        self.node_id = node_id
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.depth = depth
        self.parent_id = parent_id
        self.left_child_id: Optional[int] = None
        self.right_child_id: Optional[int] = None
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        
        # Validate bounds
        assert np.all(self.low >= DOMAIN_LOW), f"Lower bounds must be >= {DOMAIN_LOW}"
        assert np.all(self.high <= DOMAIN_HIGH), f"Upper bounds must be <= {DOMAIN_HIGH}"
        assert np.all(self.low < self.high), "Lower bounds must be < upper bounds"
        assert self.low.shape == self.high.shape, "low and high must have same shape"
        
        self.d = len(self.low)
        
        # Sample storage (only for leaves)
        self.xs: Optional[np.ndarray] = None  # Will be initialized when first sample added
        self.ys: Optional[np.ndarray] = None
    
    @property
    def diameter(self) -> float:
        """Compute the ℓ∞ diameter of the node."""
        return float(np.max(self.high - self.low))
    
    @property
    def volume(self) -> float:
        """Compute the volume of the node."""
        return float(np.prod(self.high - self.low))
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.left_child_id is None and self.right_child_id is None
    
    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is contained in this node.
        
        Args:
            x: Point to check (shape: (d,)).
            
        Returns:
            True if x is in the node (including boundaries).
        """
        x = np.asarray(x)
        return bool(np.all((x >= self.low) & (x <= self.high)))
    
    def get_longest_edge(self) -> Tuple[int, float]:
        """Get the dimension index and length of the longest edge.
        
        Returns:
            Tuple of (dimension_index, edge_length).
            Ties are broken by choosing the smallest dimension index.
        """
        edge_lengths = self.high - self.low
        max_length = np.max(edge_lengths)
        # Find first dimension with max length (tie-breaking: smallest index)
        dim_idx = int(np.argmax(edge_lengths))
        return dim_idx, float(max_length)

    
    def add_sample(self, x: np.ndarray, y: float) -> None:
        """Add a sample to this leaf node.
        
        Args:
            x: Context (shape: (d,)).
            y: Reward.
            
        Raises:
            ValueError: If node is not a leaf.
        """
        if not self.is_leaf():
            raise ValueError(f"Cannot add sample to non-leaf node {self.node_id}")
        
        x = np.asarray(x).reshape(1, -1)
        y = np.array([y])
        
        if self.xs is None:
            self.xs = x
            self.ys = y
        else:
            self.xs = np.vstack([self.xs, x])
            self.ys = np.concatenate([self.ys, y])
    
    def get_count(self) -> int:
        """Get number of samples in this node.
        
        Returns:
            Sample count (0 if no samples).
        """
        if self.xs is None:
            return 0
        return len(self.xs)
    
    def get_mean(self) -> float:
        """Get empirical mean reward.
        
        Returns:
            Mean reward (0.0 if no samples).
        """
        if self.xs is None or len(self.ys) == 0:
            return 0.0
        return float(np.mean(self.ys))
    
    def get_sum(self) -> float:
        """Get sum of rewards.
        
        Returns:
            Sum of rewards.
        """
        if self.xs is None or len(self.ys) == 0:
            return 0.0
        return float(np.sum(self.ys))
    
    def split(
        self,
        left_id: int,
        right_id: int,
        splitter: Optional[object] = None,
    ) -> Tuple["TreeNode", "TreeNode"]:
        """Split this node using a splitter or default max-edge bisection.
        
        Redistributes all samples to the appropriate child based on location.
        The splitter can use any criterion (e.g., min MSE) to determine the best split.
        
        Args:
            left_id: ID for the left child node.
            right_id: ID for the right child node.
            splitter: Optional splitter object. If None, uses max-edge bisection.
            
        Returns:
            Tuple of (left_node, right_node) with samples redistributed.
            
        Raises:
            ValueError: If node is not a leaf.
        """
        if not self.is_leaf():
            raise ValueError(f"Cannot split non-leaf node {self.node_id}")
        
        # Use splitter if provided, otherwise use max-edge bisection
        if splitter is not None:
            # Prepare inputs for splitter
            X = self.xs if self.xs is not None else np.empty((0, self.d))
            y = self.ys if self.ys is not None else np.empty(0)
            X_range = np.array([self.low, self.high])
            
            
            # Call splitter (may require y for criterion-based splitters)
            try:
                split_dim_vec, split_point_vec = splitter(X, X_range, y)
            except TypeError:
                # Some splitters may not need all parameters
                try:
                    split_dim_vec, split_point_vec = splitter(X, X_range, y)
                except TypeError:
                    # Fallback for splitters that don't need y
                    split_dim_vec, split_point_vec = splitter(X, X_range)
            
            if len(split_dim_vec) > 0:
                dim_idx = split_dim_vec[0]
                split_point = split_point_vec[0]
            else:
                # Fallback to max-edge
                dim_idx, _ = self.get_longest_edge()
                split_point = (self.low[dim_idx] + self.high[dim_idx]) / 2.0
        else:
            # Default: max-edge bisection
            dim_idx, _ = self.get_longest_edge()
            split_point = (self.low[dim_idx] + self.high[dim_idx]) / 2.0
        
        # Create left child: [low, split_point] on dimension dim_idx
        left_low = self.low.copy()
        left_high = self.high.copy()
        left_high[dim_idx] = split_point
        
        left_node = TreeNode(
            node_id=left_id,
            low=left_low,
            high=left_high,
            depth=self.depth + 1,
            parent_id=self.node_id,
        )
        
        # Create right child: [split_point, high] on dimension dim_idx
        right_low = self.low.copy()
        right_high = self.high.copy()
        right_low[dim_idx] = split_point
        
        right_node = TreeNode(
            node_id=right_id,
            low=right_low,
            high=right_high,
            depth=self.depth + 1,
            parent_id=self.node_id,
        )
        
        # Update parent node
        self.left_child_id = left_id
        self.right_child_id = right_id
        self.feature = dim_idx
        self.threshold = split_point
        
        # Redistribute samples to children
        if self.xs is not None and len(self.xs) > 0:
            # Determine which child each sample belongs to
            x_dim = self.xs[:, dim_idx]
            left_mask = x_dim < split_point
            right_mask = x_dim >= split_point
            
            # Handle boundary case (exactly at split point) - assign to left
            boundary_mask = x_dim == split_point
            left_mask = left_mask | boundary_mask
            right_mask = right_mask & ~boundary_mask
            
            # Distribute samples
            if np.any(left_mask):
                left_node.xs = self.xs[left_mask]
                left_node.ys = self.ys[left_mask]
            
            if np.any(right_mask):
                right_node.xs = self.xs[right_mask]
                right_node.ys = self.ys[right_mask]
        
        # Clear samples from parent (it's no longer a leaf)
        self.xs = None
        self.ys = None
        
        return left_node, right_node
    
    def route(self, x: np.ndarray) -> int:
        """Route a point to the appropriate child node.
        
        Args:
            x: Point to route (shape: (d,)).
            
        Returns:
            Child node ID.
            
        Raises:
            ValueError: If node is a leaf (cannot route).
        """
        if self.is_leaf():
            raise ValueError(f"Cannot route from leaf node {self.node_id}")
        
        x = np.asarray(x)
        if x[self.feature] < self.threshold:
            return self.left_child_id
        else:
            return self.right_child_id
