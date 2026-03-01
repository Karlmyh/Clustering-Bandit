"""Random number generator utilities for deterministic experiments."""

import numpy as np
from typing import Optional


def create_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a numpy random generator from a seed.
    
    Args:
        seed: Random seed. If None, uses system entropy.
        
    Returns:
        A numpy random generator.
    """
    return np.random.default_rng(seed)
