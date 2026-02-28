# Phase 3: Reward Signal & Curriculum Design

**Goal:** Guide the agent to learn complex strategies beyond simple material capturing.

## 1. Reward Decay (Low-Hanging Fruit)

**Current State:**
Fixed material reward weight (e.g., 1.0).

**Proposal:**
Linearly decay the material reward coefficient over time.

**Action Plan:**

1.  Start with a high material reward weight (e.g., 1.0) to help the agent learn basic piece values.
2.  Decay this weight to 0.0 or a very small value over the first 20-30% of training.
3.  This forces the agent to rely on the sparse game outcome reward (Win/Loss) for long-term strategy.

**Expected Gain:**
Prevents the agent from becoming a "materialist" that refuses to sacrifice pieces for checkmate.

## 2. Endgame Curriculum (Mid-Term)

**Current State:**
Training starts from the standard initial chess position.

**Proposal:**
Start training on simplified endgame positions.

**Action Plan:**

1.  Use `python-chess` or similar to generate random endgame positions (e.g., King + Queen vs King, King + Rook vs King).
2.  Train the agent to solve these endgames (achieve checkmate).
3.  Progressively add more pieces (curriculum) as the agent's win rate improves.
4.  Finally, train on full games.

**Expected Gain:**
Teaches the agent _how to checkmate_ (the ultimate goal) much faster than random exploration from the opening.

## 3. Outcome Modeling

**Current State:**
Draws might be treated as 0 or loss.

**Proposal:**
Refine the reward for draws.

**Action Plan:**

1.  Treat "Draw" as a small positive reward (e.g., +0.1) for Black and a small negative for White (or vice versa depending on perspective/rating).
2.  Alternatively, use a 3-head output for the value network (Win probability, Loss probability, Draw probability).

**Expected Gain:**
Avoids "suicide to avoid a draw" behavior in lost positions.
