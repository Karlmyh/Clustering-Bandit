# Bandit Clustering: Candidate-Set Contextual Bandits with Adaptive Partitioning

A research codebase implementing **adaptive-partition UCB** for **candidate-set nonparametric contextual bandits**, extended to a **multi-user clustering/collaboration** setting. The algorithm uses max-edge splitting rules to adaptively partition the context space and maintains bin-wise statistics for UCB decision-making.

## Quick Description

This project implements:
- **Candidate-set bandits**: At each round, the learner receives a set of K candidate contexts and must choose one, with regret measured against the best candidate in the set (candidate-oracle regret).
- **Adaptive partitioning**: The context space is recursively partitioned using a max-edge splitting rule, with bins split when they accumulate sufficient samples.
- **Multi-user clustering**: Multiple users with potentially similar reward functions can collaborate by sharing statistics within estimated clusters.

## Installation

```bash
# Clone the repository
cd Clustering_Bandit

# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Quickstart

### Single-User Experiment

```bash
python scripts/run_single.py --seed 0 --T 2000 --d 2 --K 10
```

This runs a single-user experiment with:
- T=2000 rounds
- d=2 dimensional contexts
- K=10 candidates per round

Results are written to `results/raw/` and `results/summary/`.

### Multi-User Experiment

```bash
python scripts/run_multi.py --seed 0 --T 2000 --m 10 --clusters 2 --d 2 --K 10
```

This runs a multi-user experiment with:
- T=2000 rounds
- m=10 users
- 2 clusters (5 users per cluster)
- d=2 dimensional contexts
- K=10 candidates per round

## Regret Definition

The regret is measured against the **candidate-oracle** π★, which selects the best candidate from the provided set at each round:

```
R(π) = Σ_t E_{𝒟∼P_X^K}[ f(π★(𝒟)) - f(π_t(𝒟)) | 𝓕_{t-1} ]
```

where π★(𝒟) ∈ argmax_{x∈𝒟} f(x).

This is different from global-oracle regret (vs sup_x f(x)), which would be harder to achieve. The candidate-oracle regret measures how well the algorithm performs relative to the best choice available in each candidate set.

## Project Structure

```
bandit-clustering/
├── src/
│   └── bandit_clustering/
│       ├── __init__.py
│       ├── config.py              # Configuration constants
│       ├── utils/                  # Utilities (RNG, logging)
│       ├── partition/              # Partition structure (Bin, TreePartition)
│       ├── bandits/                # Environment (CandidateEnv, reward functions)
│       ├── agents/                  # Agents (AdaptiveUCB, baselines)
│       └── multi_user/              # Multi-user clustering framework
├── tests/                           # Pytest test suite
├── scripts/                         # Experiment scripts
│   ├── run_single.py
│   ├── run_multi.py
│   ├── aggregate_results.py
│   └── make_plots.py
└── results/                         # Generated results
    ├── raw/                         # Per-run logs
    ├── summary/                     # Aggregated statistics
    └── figures/                     # Plots
```

## Reproduce Experiments

### Single-User Regret Curves

```bash
# Run multiple seeds
for seed in {0..9}; do
    python scripts/run_single.py --seed $seed --T 5000 --d 2 --K 10
done

# Aggregate results
python scripts/aggregate_results.py --pattern "results/raw/single_*"

# Generate plots
python scripts/make_plots.py --input results/summary/ --output results/figures/
```

### Multi-User Clustering Demo

```bash
python scripts/run_multi.py --seed 0 --T 5000 --m 20 --clusters 3 --d 2 --K 10
```

## Algorithm Overview

### Adaptive Partition UCB

1. **Partition Structure**: Maintains a binary tree partition of [0,1]^d into axis-aligned hyper-rectangles (bins).
2. **Max-Edge Splitting**: When a bin is split, it's bisected along its longest edge (ties broken by dimension index).
3. **Bin Statistics**: Each bin B maintains:
   - N(B): sample count
   - f̂(B): empirical mean reward
   - U(B): UCB confidence width = sqrt(2 log(T) / N(B))
4. **Splitting Rule**: Split bin B when N(B) ≥ c_split * log(T) / diam(B)^2
5. **UCB Policy**: For each candidate x, compute UCB(x) = f̂(B(x)) + U(B(x)), then select argmax.

### Multi-User Clustering

- Maintains a **shared partition** across all users
- Periodically clusters users based on empirical disagreement in high-value regions
- Shares bin statistics within estimated clusters by pooling counts and rewards

## Outputs

- **`results/raw/`**: JSONL logs per run with per-round metrics
- **`results/summary/`**: Aggregated statistics (mean regret curves, etc.)
- **`results/figures/`**: Matplotlib plots of regret curves and other visualizations

## Testing

```bash
pytest -q
```

Tests validate:
- Partition splitting correctness and geometry
- Bin statistics correctness
- UCB selection correctness
- Determinism with fixed seeds

## License

MIT
