"""
Empirical Reward Overoptimization & Policy Drift Simulator
Implements mathematical modeling of Goodhart's Law in RLHF & DPO alignment.
Tracks Proxy Reward vs Gold Alignment Reward, KL Divergence, and Length Bias Exploitation.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def run_overoptimization_simulation(
    total_steps: int = 500,
    beta_values: List[float] = [0.01, 0.05, 0.1, 0.5],
    seed: int = 42
) -> Dict[str, Any]:
    np.random.seed(seed)
    
    steps = np.arange(0, total_steps + 1, 10)
    trajectories = {}
    
    for beta in beta_values:
        # Lower beta allows policy to drift farther from reference policy (higher KL drift)
        kl_drift = (steps / 100.0) ** 1.4 * (0.1 / beta) ** 0.6
        
        # Proxy reward increases monotonically as policy optimizes proxy objective
        proxy_reward = 2.0 + 3.5 * (1.0 - np.exp(-steps / (80.0 * (beta / 0.1) ** 0.5)))
        # Add slight stochastic noise
        proxy_reward += np.random.normal(0, 0.04, size=len(steps))
        
        # True gold reward follows Gao et al. (2023) overoptimization curve:
        # Peaks early, then degrades as policy exploits flaws / length shortcuts in proxy model
        optimal_step = int(120 * (beta / 0.1) ** 0.7)
        peak_reward = 3.2
        
        gold_reward = np.zeros_like(steps, dtype=float)
        for idx, s in enumerate(steps):
            if s <= optimal_step:
                gold_reward[idx] = 2.0 + (peak_reward - 2.0) * (s / optimal_step) ** 0.85
            else:
                # Degradation regime (reward hacking / overoptimization)
                decay_rate = 0.0035 * (0.1 / beta) ** 0.5
                gold_reward[idx] = peak_reward * np.exp(-decay_rate * (s - optimal_step))
        gold_reward += np.random.normal(0, 0.03, size=len(steps))
        
        # Length inflation (character count)
        base_len = 220
        length_traj = base_len + (steps / total_steps) * 450 * (0.1 / beta) ** 0.5
        
        trajectories[f"beta_{beta}"] = {
            "steps": steps.tolist(),
            "kl_divergence": np.round(kl_drift, 3).tolist(),
            "proxy_reward": np.round(proxy_reward, 3).tolist(),
            "gold_reward": np.round(gold_reward, 3).tolist(),
            "average_response_length": np.round(length_traj, 1).tolist()
        }
        
    return trajectories

if __name__ == "__main__":
    trajectories = run_overoptimization_simulation()
    
    with open("/Users/authrain/development/research/rlhf-reward-overoptimization-lab/results/overoptimization_trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)
        
    # Create tabular summary of peak vs final gold reward
    summary = []
    for beta_key, data in trajectories.items():
        gold = data["gold_reward"]
        proxy = data["proxy_reward"]
        kl = data["kl_divergence"]
        lens = data["average_response_length"]
        
        peak_idx = int(np.argmax(gold))
        summary.append({
            "KL_Penalty_Beta": float(beta_key.replace("beta_", "")),
            "Peak_Gold_Reward": float(np.max(gold)),
            "Step_at_Peak": int(data["steps"][peak_idx]),
            "Final_Gold_Reward": float(gold[-1]),
            "Final_Proxy_Reward": float(proxy[-1]),
            "Reward_Overoptimization_Drop_Pct": round(float((np.max(gold) - gold[-1]) / np.max(gold) * 100), 2),
            "Final_KL_Divergence": float(kl[-1]),
            "Length_Inflation_Factor": round(float(lens[-1] / lens[0]), 2)
        })
        
    df_summary = pd.DataFrame(summary).sort_values(by="KL_Penalty_Beta")
    df_summary.to_csv("/Users/authrain/development/research/rlhf-reward-overoptimization-lab/results/summary_table.csv", index=False)
    print("\n--- REWARD OVEROPTIMIZATION DYNAMICS SUMMARY ---")
    print(df_summary.to_markdown(index=False))
