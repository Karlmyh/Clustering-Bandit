"""Splitter classes for determining how to split nodes.

Following reference_code/_splitter.py pattern.
"""

import numpy as np
from typing import List, Optional, Tuple

from .criterion import mse

class MaxEdgeRandomSplitter:
    """Max edge random splitter class.
    
    Randomly selects a dimension with longest edge and split point.
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, X: np.ndarray, X_range: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Determine split dimension and point.
        
        Args:
            X: Array of points (shape: (n, d)).
            X_range: Node boundaries (shape: (2, d)) - [low, high].
            y: Array of labels (shape: (n,)).   
        """
        edge_ratio = X_range[1] - X_range[0]
        max_length = np.max([edge_ratio[i] for i in range(len(edge_ratio)) ])
        max_edges = np.where(edge_ratio == max_length)[0]
        split_dim_vec = []
        split_point_vec = []
        for rd_dim in max_edges:
            split_dim_vec.append(rd_dim)
            split_point_vec.append((X_range[1, rd_dim] + X_range[0, rd_dim]) / 2)
        return split_dim_vec, split_point_vec

class GainReductionSplitter:
    """Variance reduction based splitter class.
    
    Searches across all dimensions (not just max-edge) to find the best split
    using MSE or ANLL criterion. This is the general decision tree partition rule.
    """

    def __init__(self, criterion: str = "mse", search_number: int = 10, rng: Optional[np.random.Generator] = None):
        """Initialize splitter.
        
        Args:
            criterion: Criterion to use ("mse" or "anll").
            search_number: Number of candidate split points to evaluate per dimension.
            rng: Random number generator.
        """
        self.search_number = search_number
        self.compute_criterion_reduction = mse
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        X: np.ndarray,
        X_range: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[List[int], List[float]]:
        """Determine split dimension and point.
        
        Searches across all dimensions to find the best split point that maximizes
        criterion reduction (e.g., minimizes MSE).
        
        Args:
            X: Array of points (shape: (n, d)).
            X_range: Node boundaries (shape: (2, d)) - [low, high].
            y: Array of labels (shape: (n,)).
            
        Returns:
            Tuple of (split_dim_list, split_point_list).
        """

        n_node_samples, dim = X.shape
        
        split_dim_vec = []
        split_point_vec = []
        criterion_vec = []
        
        # Search for dimension and split point with maximum criterion reduction
        for d in range(dim):
            
            # Get unique values in this dimension
            dt_X_dim_unique = np.unique(X[:, d])
            
            # Generate candidate split points using quantiles
            if len(dt_X_dim_unique) > 0:
                sorted_split_point = np.unique(
                    np.quantile(
                        dt_X_dim_unique,
                        [(2 * i + 1) / (2 * self.search_number) for i in range(self.search_number)]
                    )
                )
            else:
                # Fallback to midpoint if no unique values
                sorted_split_point = np.array([(X_range[0, d] + X_range[1, d]) / 2])
            
            split_point = None
            max_criterion_reduction = -np.inf
            
            for split in sorted_split_point:
                criterion_reduction = self.compute_criterion_reduction(
                    X, X_range, y, d, split
                )
                if criterion_reduction > max_criterion_reduction:
                    max_criterion_reduction = criterion_reduction
                    split_point = split
            
            split_dim_vec.append(d)
            split_point_vec.append(split_point)
            criterion_vec.append(max_criterion_reduction)

        # Sort by criterion reduction and take top candidates
        sorted_indices = sorted(
            range(len(criterion_vec)), key=lambda i: criterion_vec[i], reverse=True
        )
        ratio_of_dims_totake = max(1, (len(sorted_indices) + 5) // 10)
        sorted_indices = sorted_indices[0:ratio_of_dims_totake]
        sorted_split_point = [split_point_vec[i] for i in sorted_indices]
        sorted_split_dim = [split_dim_vec[i] for i in sorted_indices]

        return sorted_split_dim, sorted_split_point


class GainReductionMaxEdgeSplitter:
    """Variance reduction based max edge splitter class.
    
    Considers only dimensions with longest edge, evaluates splits using MSE criterion,
    and returns the best split(s).
    """

    def __init__(self, criterion: str = "mse", search_number: int = 10, rng: Optional[np.random.Generator] = None):
        """Initialize splitter.
        
        Args:
            criterion: Criterion to use ("mse" or "anll").
            search_number: Number of candidate split points to evaluate.
            rng: Random number generator.
        """
        self.search_number = search_number
        self.compute_criterion_reduction = mse
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        X: np.ndarray,
        X_range: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[List[int], List[float]]:
        """Determine split dimension and point.
        
        Args:
            X: Array of points (shape: (n, d)).
            X_range: Node boundaries (shape: (2, d)) - [low, high].
            y: Array of labels (shape: (n,)).
            
        Returns:
            Tuple of (split_dim_list, split_point_list).
        """

        edge_ratio = X_range[1] - X_range[0]
        max_length = np.max([edge_ratio[i] for i in range(len(edge_ratio)) ])
        max_edges = np.where(edge_ratio == max_length)[0]

    
        if len(max_edges) == 0:
            return [], []
        
        split_dim_vec = []
        split_point_vec = []
        criterion_vec = []

        # Search for dimension with maximum criterion reduction
        for rd_dim in max_edges:
            split = (X_range[1, rd_dim] + X_range[0, rd_dim]) / 2
            split_dim_vec.append(rd_dim)
            split_point_vec.append(split)
            criterion_vec.append(self.compute_criterion_reduction(X, X_range, y, rd_dim, split))

        sorted_indices = sorted(range(len(criterion_vec)), key=lambda i: criterion_vec[i], reverse=True)
        ratio_of_dims_totake = max(1, (len(sorted_indices) + 5) // 10)
        sorted_indices = sorted_indices[0:ratio_of_dims_totake]
        sorted_split_point = [split_point_vec[i] for i in sorted_indices]
        sorted_split_dim = [split_dim_vec[i] for i in sorted_indices]

        return sorted_split_dim, sorted_split_point
