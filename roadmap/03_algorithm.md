# Phase 2: Algorithmic Paradigms

**Goal:** Move from basic A2C to more stable and powerful RL algorithms.

## 1. PPO + GAE (Immediate Priority)

**Current State:**
Synchronous A2C-style updates.

**Proposal:**
Upgrade to **Proximal Policy Optimization (PPO)** with **Generalized Advantage Estimation (GAE)**.

**Action Plan:**

1.  **GAE Implementation:**
    - Modify `backpropagate` to calculate advantages using GAE-Lambda.
    - This reduces variance in gradient estimates.
2.  **PPO Clipping:**
    - Implement the PPO clipped objective function for the policy loss.
    - This prevents destructively large policy updates and improves training stability.
3.  **Mini-batch Updates:**
    - Instead of updating on the entire rollout at once, shuffle the collected data and update on mini-batches for multiple epochs.

**Expected Gain:**
Significantly improved training stability and sample efficiency.

## 2. MCTS at Inference (Mid-Term)

**Current State:**
Policy network plays directly (greedy or sampled).

**Proposal:**
Use **Monte Carlo Tree Search (MCTS)** during evaluation/inference to boost playing strength.

**Action Plan:**

1.  Implement a basic MCTS that uses the policy network for prior probabilities and the value network for leaf evaluation.
2.  Use this MCTS only during evaluation games (e.g., against Stockfish) to improve decision quality.
3.  Keep training model-free (PPO) for now to avoid the high computational cost of AlphaZero-style training.

**Expected Gain:**
Higher playing strength without the massive training cost of AlphaZero.
