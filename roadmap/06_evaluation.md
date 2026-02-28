# Phase 5: Evaluation Metrics

**Goal:** Accurately measure playing strength, as Loss is not a good proxy for ELO.

## 1. Stockfish Evaluation Script (Immediate Priority)

**Current State:**
Monitoring Loss and Entropy.

**Proposal:**
Create a separate script to evaluate the agent against a fixed baseline (Stockfish).

**Action Plan:**

1.  Create `evaluate.py`.
2.  Implement a loop that plays N games (e.g., 20) between the current agent and Stockfish.
3.  Limit Stockfish strength (e.g., `Skill Level 0` or limited depth/nodes) to make it a beatable benchmark for early training.
4.  Report the Win/Draw/Loss rate.
5.  Run this evaluation periodically (e.g., every 1000 training steps).

**Expected Gain:**
Provides a true, objective measure of the agent's progress and playing strength.

## 2. Puzzle Accuracy (Mid-Term)

**Current State:**
No tactical evaluation.

**Proposal:**
Evaluate the policy network on a dataset of chess puzzles.

**Action Plan:**

1.  Curate a small dataset of "mate-in-1" or simple tactical puzzles (FEN strings + correct moves).
2.  Periodically run the policy network on these positions.
3.  Measure the accuracy (percentage of times the correct move is the top-ranked action).

**Expected Gain:**
Indicates if the model is learning actual chess tactics or just overfitting to self-play dynamics.
