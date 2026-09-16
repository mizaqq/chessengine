# Phase 5: Evaluation Metrics

**Goal:** Accurately measure playing strength, as Loss is not a good proxy for ELO.

## 0. Mate-in-One Rate (Adopted 2026-09-15)

**Metric:** over N sampled self-play games, share of positions with a mate-in-one
available where the mover played a mate. Baseline (scale-0.2 checkpoints, 40 games):
8 / 215 = 3.7%. Cheap, needs no engine, directly measures the weakness behind the
draw rate. Add to the metrics watched in every algorithm/curriculum change.
Evaluation must sample (low temperature), not argmax: greedy play loops into
threefold repetition.

**Fixed benchmark (2026-09-16):** the in-game rate depends on which positions
self-play reaches, so the primary metric is now the held-out Lichess set
(`data/puzzles/mate_in_1_eval.csv`, 2,000 positions): top-1 accuracy and mean
mating-move probability, `scripts/eval_checkpoint_puzzles.py`. Scale-0.2
checkpoints: top-1 7.35%, probability 0.065. In-game rate stays as the transfer
check (`scripts/eval_games.py`). This supersedes §2 below.

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
