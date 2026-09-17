"""
Publication Figure Generator for RLHF Reward Overoptimization Lab
Generates:
1. figures/overoptimization_curves.png: Proxy vs Gold Alignment Reward trajectory
2. figures/length_exploitation_drift.png: Length inflation vs KL drift
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, sans-serif'
plt.rcParams['font.size'] = 10

def plot_overoptimization():
    with open("/Users/authrain/development/research/rlhf-reward-overoptimization-lab/results/overoptimization_trajectories.json") as f:
        data = json.load(f)
        
    steps = data["beta_0.1"]["steps"]
    proxy = data["beta_0.1"]["proxy_reward"]
    gold = data["beta_0.1"]["gold_reward"]
    kl = data["beta_0.1"]["kl_divergence"]
    
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    
    # Left axis: Rewards
    color_proxy = "#2563eb"
    color_gold = "#dc2626"
    
    l1 = ax1.plot(steps, proxy, color=color_proxy, lw=2.5, label="Proxy Reward Model (r_hat)")
    l2 = ax1.plot(steps, gold, color=color_gold, lw=2.5, label="True Gold Alignment Reward (r*)")
    ax1.set_xlabel("Policy Optimization Steps (RL / DPO Iterations)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Reward Value", fontsize=11, fontweight='bold')
    ax1.set_ylim(1.5, 6.0)
    
    # Inflection annotation
    peak_idx = int(np.argmax(gold))
    peak_step = steps[peak_idx]
    peak_val = gold[peak_idx]
    ax1.axvline(x=peak_step, color='gray', linestyle=':', lw=1.5, alpha=0.8)
    ax1.annotate(f"Overoptimization Inflection Point\n(Goodhart's Law Boundary, step {peak_step})",
                 xy=(peak_step, peak_val), xytext=(peak_step + 40, peak_val + 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontweight='medium', bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef3c7', edgecolor='#d97706'))
                 
    # Right axis: KL Divergence
    ax2 = ax1.twinx()
    color_kl = "#059669"
    l3 = ax2.plot(steps, kl, color=color_kl, lw=2, linestyle='--', label="KL Divergence D_KL(pi_theta || pi_ref)")
    ax2.set_ylabel("KL Divergence (nats)", color=color_kl, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_kl)
    ax2.set_ylim(0, 15)
    ax2.grid(False)
    
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)
    
    plt.title("Reward Model Overoptimization in Preference Fine-Tuning (β = 0.1)", fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig("/Users/authrain/development/research/rlhf-reward-overoptimization-lab/figures/overoptimization_curves.png", dpi=300)
    plt.close()
    print("Saved overoptimization_curves.png")

def plot_length_exploitation():
    with open("/Users/authrain/development/research/rlhf-reward-overoptimization-lab/results/overoptimization_trajectories.json") as f:
        data = json.load(f)
        
    betas = [0.01, 0.05, 0.1, 0.5]
    colors = ["#dc2626", "#ea580c", "#2563eb", "#059669"]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    for b, col in zip(betas, colors):
        kl = data[f"beta_{b}"]["kl_divergence"]
        lens = data[f"beta_{b}"]["average_response_length"]
        ax.plot(kl, lens, lw=2.2, color=col, label=f"KL Penalty β = {b}")
        
    ax.set_title("Response Length Exploitation as Policy Drifts from Reference", fontweight='bold', fontsize=12)
    ax.set_xlabel("KL Divergence from Reference Policy D_KL(pi_theta || pi_ref) [nats]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean Response Token Length (Chars)", fontsize=11, fontweight='bold')
    ax.legend(title="Regularization", frameon=True)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("/Users/authrain/development/research/rlhf-reward-overoptimization-lab/figures/length_exploitation_drift.png", dpi=300)
    plt.close()
    print("Saved length_exploitation_drift.png")

if __name__ == "__main__":
    plot_overoptimization()
    plot_length_exploitation()
