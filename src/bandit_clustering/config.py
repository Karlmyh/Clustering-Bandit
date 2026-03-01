"""Configuration constants for the bandit clustering algorithm."""

# Domain configuration
DOMAIN_LOW = 0.0  # Lower bound for all dimensions (default: [0,1]^d)
DOMAIN_HIGH = 1.0  # Upper bound for all dimensions (default: [0,1]^d)

# UCB confidence parameter
UCB_CONFIDENCE_MULTIPLIER = 2.0  # sqrt(UCB_CONFIDENCE_MULTIPLIER * log(T) / N(B))
APPROXIMATION_ERROR_FACTOR = 1.0

# Splitting rule constant
C_SPLIT = 1.0  # Split when N(B) >= c_split * log(T) / diam(B)^2

# Multi-user clustering
CLUSTERING_UPDATE_INTERVAL = 200  # Update clusters every M rounds
CLUSTERING_DISAGREEMENT_THRESHOLD = 0.1  # Threshold for clustering users

# Default experiment parameters
DEFAULT_T = 2000
DEFAULT_D = 2
DEFAULT_K = 10
DEFAULT_M = 10
DEFAULT_NUM_CLUSTERS = 2
