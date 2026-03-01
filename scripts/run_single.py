#!/usr/bin/env python3
"""Run a single-user bandit experiment."""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bandit_clustering.bandits import CandidateSetBanditEnv
from src.bandit_clustering.bandits.reward_functions import LinearReward
from src.bandit_clustering.agents import AdaptivePartitionUCB, BinnedPartitionUCB
from src.bandit_clustering.utils.logging import save_jsonl
from src.bandit_clustering.utils.rng import create_rng



def main():
    parser = argparse.ArgumentParser(description="Run single-user bandit experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--T", type=int, default=512000, help="Number of rounds")
    parser.add_argument("--d", type=int, default=1, help="Dimension")
    parser.add_argument("--K", type=int, default=32, help="Number of candidates per round")
    parser.add_argument("--reward", type=str, default="linear", choices=["linear"])
    parser.add_argument("--agent", type=str, default="binucb", choices=["adaucb", "binucb"])
    parser.add_argument("--depth", type=int, default=6, help="Depth for binned partition (only for binucb)")
    parser.add_argument("--output-dir", type=str, default="results/raw", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize RNG
    rng = create_rng(args.seed)
    
    # Create reward function
    if args.reward == "linear":
        reward_fn = LinearReward(d=args.d, rng=rng)
    
    # Create environment
    env = CandidateSetBanditEnv(reward_fn=reward_fn, K=args.K, seed=args.seed)
    
    # Create agent
    if args.agent == "adaucb":
        agent = AdaptivePartitionUCB(d=args.d, T=args.T, seed=args.seed)
    elif args.agent == "binucb":
        agent = BinnedPartitionUCB(d=args.d, T=args.T, depth=args.depth, seed=args.seed)
    
    # Run experiment
    logs = []
    cumulative_regret = 0.0
    cumulative_global_regret = 0.0
    
    for t in range(args.T):
        # Generate candidates
        candidates = env._generate_candidates()
        
        # Select action
        action_idx = agent.select_action(candidates, t)
        
        # Step environment
        selected_candidate, noisy_reward = env.step(candidates, action_idx)
        info = env.history[-1]
    
        agent.update(selected_candidate, noisy_reward)
        agent.perform_splits()

        cumulative_regret += info["regret"]
        cumulative_global_regret += info["global_regret"]

        
        
    # print([single_history["optimal_reward"] for single_history in env.history])
    # print([single_history["oracle_reward"] for single_history in env.history])
    # print([single_history["selected_reward"] for single_history in env.history])
    # print([single_history["regret"] for single_history in env.history])
    # print([single_history["global_regret"] for single_history in env.history])
        if t > 3000:
            if t % 3000 == 0:
                # plot reward_function and ucbs
                X = np.linspace(0, 1, 100).reshape(-1, 1)
                plt.plot(X, reward_fn(X), label="reward_function")
                plt.plot(X, agent.get_ucbs(X), label="ucbs")
                samples = [hist["selected_candidate"] for hist in env.history]
                rewards = [hist["noisy_reward"] for hist in env.history]

                plt.scatter(samples, rewards, label="samples", color="red", marker="x", s=1)
                plt.legend()
                plt.savefig(output_dir / f"reward_function_and_ucbs_seed{args.seed}_T{args.T}_d{args.d}_K{args.K}_{args.reward}_{args.agent}_t{t}.png")
                plt.close()

                
                # widths = agent.get_ucb_widths(np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)) 
                # print(widths)
        else:
            if t % 300 == 0:
                # plot reward_function and ucbs
                X = np.linspace(0, 1, 100).reshape(-1, 1)
                plt.plot(X, reward_fn(X), label="reward_function")
                plt.plot(X, agent.get_ucbs(X), label="ucbs")
                samples = [hist["selected_candidate"] for hist in env.history]
                rewards = [hist["noisy_reward"] for hist in env.history]

                plt.scatter(samples, rewards, label="samples", color="red", marker="x", s=1)
                plt.legend()
                plt.savefig(output_dir / f"reward_function_and_ucbs_seed{args.seed}_T{args.T}_d{args.d}_K{args.K}_{args.reward}_{args.agent}_t{t}.png")
                plt.close()

               
                # widths = agent.get_ucb_widths(np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)) 
                # print(widths)
                
    
    print("K", args.K)
    print("cumulative_regret", cumulative_regret)
    print("cumulative_global_regret", cumulative_global_regret)




if __name__ == "__main__":
    main()
