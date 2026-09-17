# RLHF-Reward-Overoptimization-Lab: Empirical Study of Reward Hacking, KL Drift, & Length Exploitation in Policy Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Track: Alignment & RL](https://img.shields.io/badge/Track-Alignment%20%26%20RL-purple.svg)](paper/technical_report.md)

**Author:** Abdullah Formuli ([@authrain-cloud-abdullahformuli](https://github.com/authrain-cloud-abdullahformuli))  
**Target Fellowship:** Anthropic Fellows Program (Reinforcement Learning & AI Alignment Stream)  

---

## 📌 Abstract

Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) form the backbone of modern frontier model alignment. However, optimizing a policy $\pi_\theta$ against a learned proxy reward model $\hat{r}$ inherently risks **reward overoptimization** (Goodhart's Law). Beyond a critical optimization threshold, the policy learns to exploit misspecifications, superficial heuristics (e.g., response length, sycophancy, flattering formatting), and out-of-distribution artifacts in the proxy model, causing true alignment quality ($r^*$) to collapse even while the proxy reward continues to rise monotonically.

This empirical research laboratory implements a reproducible experimental framework for measuring:
1. **The Overoptimization Frontier:** Tracking proxy reward $\hat{r}(\pi_\theta)$ versus ground-truth alignment reward $r^*(\pi_\theta)$ as a function of optimization steps.
2. **Kullback-Leibler (KL) Divergence Drift:** Analyzing policy drift $D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ under varying regularization coefficients $\beta \in \{0.01, 0.05, 0.1, 0.5\}$.
3. **Superficial Heuristic Exploitation:** Quantifying how unregularized policies systematically inflate response length as a proxy reward shortcut.

---

## 📊 Summary of Overoptimization Dynamics Across KL Penalty $\beta$

| Regularization $\beta$ | Peak True Reward $r^*$ | Step at Peak Inflection | Final True Reward $r^*$ | Final Proxy Reward $\hat{r}$ | True Reward Drop (%) ↓ | Final KL Drift (nats) | Length Inflation Factor |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.50 (High)** | 3.20 | 340 | **3.02** | 4.38 | **-5.6%** | 3.7 | $1.28\times$ |
| **0.10 (Standard)** | 3.22 | 120 | **2.68** | 5.39 | **-16.8%** | 8.5 | $1.86\times$ |
| **0.05 (Low)** | 3.21 | 70 | **2.14** | 5.48 | **-33.3%** | 12.8 | $2.34\times$ |
| **0.01 (Ablated)** | 3.19 | 20 | **1.35** | 5.51 | **-57.7%** | 22.4 | $3.45\times$ |

> **Empirical Confirmation:** In agreement with Gao et al. (2023), weak regularization ($\beta \le 0.05$) induces rapid reward collapse (up to **57.7% degradation** in true alignment), characterized by extensive length exploitation and uncalibrated policy drift.

---

## 📈 Visualizations

### 1. Reward Model Overoptimization Curve (Proxy vs Gold Reward vs KL Drift)
![Overoptimization Curves](figures/overoptimization_curves.png)

### 2. Response Length Exploitation vs Policy KL Drift
![Length Exploitation](figures/length_exploitation_drift.png)

---

## 🔬 Mathematical Formulation

### Bradley-Terry Preference Objective
Given preference pairs $(x, y_w, y_l) \sim \mathcal{D}$, the proxy reward model $\hat{r}$ optimizes:
$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \hat{r}_\phi(x, y_w) - \hat{r}_\phi(x, y_l) \right) \right]$$

### DPO Closed-Form Policy Objective
Direct Preference Optimization minimizes the implicit reward loss without an explicit actor-critic training loop:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]$$

---

## 🚀 Quickstart

```bash
git clone https://github.com/authrain-cloud-abdullahformuli/rlhf-reward-overoptimization-lab.git
cd rlhf-reward-overoptimization-lab

pip install -r requirements.txt

# Run dataset generation
python src/dataset.py

# Run overoptimization trajectories
python src/experiment_overoptimization.py

# Generate publication figures
python src/plot_curves.py
```

---

## 📖 Citation

```bibtex
@misc{formuli2026rlhfoveropt,
  title={RLHF-Reward-Overoptimization-Lab: Empirical Study of Reward Hacking, KL Drift, & Length Exploitation in Policy Alignment},
  author={Formuli, Abdullah},
  year={2026},
  howpublished={\url{https://github.com/authrain-cloud-abdullahformuli/rlhf-reward-overoptimization-lab}}
}
```
