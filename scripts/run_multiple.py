#!/usr/bin/env python3
"""Run multiple single-user experiments and save regret curves to one CSV."""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bandit_clustering.agents import AdaptivePartitionUCB, BinnedPartitionUCB
from src.bandit_clustering.bandits import CandidateSetBanditEnv
from src.bandit_clustering.bandits.reward_functions import LinearReward, QuadraticReward
from src.bandit_clustering.utils.rng import create_rng


def parse_int_list(raw: str) -> List[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a comma-separated list of integers.")
    return [int(v) for v in values]


def build_reward_fn(name: str, d: int, rng):
    if name == "linear":
        return LinearReward(d=d, rng=rng)
    if name == "quadratic":
        return QuadraticReward(d=d, rng=rng)
    raise ValueError(f"Unsupported reward function: {name}")


def run_one_configuration(
    *,
    seed: int,
    K: int,
    d: int,
    T: int,
    record_every: int,
    reward_name: str,
    agent_name: str,
    depth: Optional[int],
) -> List[Tuple[int, float, float]]:
    rng = create_rng(seed)
    reward_fn = build_reward_fn(reward_name, d=d, rng=rng)
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=K, seed=seed)

    if agent_name == "adaucb":
        agent = AdaptivePartitionUCB(d=d, T=T, seed=seed)
    elif agent_name == "binucb":
        if depth is None:
            raise ValueError("Depth is required for binucb.")
        agent = BinnedPartitionUCB(d=d, T=T, depth=depth, seed=seed)
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    checkpoints = set(range(record_every, T + 1, record_every))
    series: List[Tuple[int, float, float]] = [(0, 0.0, 0.0)]

    cumulative_regret = 0.0
    cumulative_global_regret = 0.0

    for t in range(T):
        candidates = env._generate_candidates()
        action_idx = agent.select_action(candidates, t)
        selected_candidate, noisy_reward = env.step(candidates, action_idx)
        info = env.history[-1]

        agent.update(selected_candidate, noisy_reward)
        agent.perform_splits()

        cumulative_regret += float(info["regret"])
        cumulative_global_regret += float(info["global_regret"])

        round_idx = t + 1
        if round_idx in checkpoints:
            series.append((round_idx, cumulative_regret, cumulative_global_regret))

    return series


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple regret experiments to one CSV.")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed (inclusive).")
    parser.add_argument("--num-runs", type=int, default=30, help="Number of repeated runs.")
    parser.add_argument("--T", type=int, default=120000, help="Total number of rounds.")
    parser.add_argument("--record-every", type=int, default=10000, help="Record interval in rounds.")
    parser.add_argument("--d", type=int, default=1, help="Context dimension.")
    parser.add_argument("--Ks", type=str, default="2,4,8,16,32", help="Comma-separated K values.")
    parser.add_argument(
        "--depths",
        type=str,
        default="2,4,6",
        help="Comma-separated depth values for binucb.",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="linear",
        choices=["linear", "quadratic"],
        help="Reward function.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/summary/multiple_regret.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.T <= 0:
        raise ValueError("--T must be positive.")
    if args.record_every <= 0:
        raise ValueError("--record-every must be positive.")
    if args.T % args.record_every != 0:
        raise ValueError("--T must be divisible by --record-every to include the final checkpoint.")

    Ks = parse_int_list(args.Ks)
    depths = parse_int_list(args.depths)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "seed",
        "K",
        "d",
        "reward",
        "agent",
        "depth",
        "T",
        "round",
        "cumulative_regret",
        "cumulative_global_regret",
    ]

    run_counter = 0
    total_jobs = args.num_runs * len(Ks) * (1 + len(depths))

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_offset in range(args.num_runs):
            seed = args.seed_start + run_offset
            for K in Ks:
                run_counter += 1
                print(f"[{run_counter}/{total_jobs}] seed={seed} K={K} agent=adaucb")
                adaucb_series = run_one_configuration(
                    seed=seed,
                    K=K,
                    d=args.d,
                    T=args.T,
                    record_every=args.record_every,
                    reward_name=args.reward,
                    agent_name="adaucb",
                    depth=None,
                )
                for round_idx, cumulative_regret, cumulative_global_regret in adaucb_series:
                    writer.writerow(
                        {
                            "run_id": run_offset,
                            "seed": seed,
                            "K": K,
                            "d": args.d,
                            "reward": args.reward,
                            "agent": "adaucb",
                            "depth": "",
                            "T": args.T,
                            "round": round_idx,
                            "cumulative_regret": cumulative_regret,
                            "cumulative_global_regret": cumulative_global_regret,
                        }
                    )

                for depth in depths:
                    run_counter += 1
                    print(f"[{run_counter}/{total_jobs}] seed={seed} K={K} agent=binucb depth={depth}")
                    binucb_series = run_one_configuration(
                        seed=seed,
                        K=K,
                        d=args.d,
                        T=args.T,
                        record_every=args.record_every,
                        reward_name=args.reward,
                        agent_name="binucb",
                        depth=depth,
                    )
                    for round_idx, cumulative_regret, cumulative_global_regret in binucb_series:
                        writer.writerow(
                            {
                                "run_id": run_offset,
                                "seed": seed,
                                "K": K,
                                "d": args.d,
                                "reward": args.reward,
                                "agent": "binucb",
                                "depth": depth,
                                "T": args.T,
                                "round": round_idx,
                                "cumulative_regret": cumulative_regret,
                                "cumulative_global_regret": cumulative_global_regret,
                            }
                        )

    print(f"Saved CSV to: {output_csv}")


if __name__ == "__main__":
    main()
