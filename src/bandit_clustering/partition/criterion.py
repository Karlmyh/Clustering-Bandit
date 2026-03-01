"""Criterion functions for evaluating splits.

Following reference_code/_criterion.py pattern.
"""

import numpy as np


def insample_ssq(y: np.ndarray) -> float:
    """In-sample sum of squares.
    
    Args:
        y: Array of values.
        
    Returns:
        Sum of squares.
    """
    if len(y) > 0:
        return float(np.var(y) * len(y))
    return 0.0


def mse(X: np.ndarray, X_range: np.ndarray, Y: np.ndarray, d: int, split: float) -> float:
    """Compute MSE decrease for one pair of dimension and split point.
    
    Parameters
    ----------
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
 
    d : int in 0, ..., dim - 1
        The splitting dimension.
    
    split : float
        The splitting point.
    
    Returns
    -------
    mse : float
        Negative sum of squared errors (higher is better).
    """
    x = X[:, d]
    y = Y

    if len(x) == 0:
        return -np.inf
    return float(-insample_ssq(y[x < split]) - insample_ssq(y[x >= split]))

