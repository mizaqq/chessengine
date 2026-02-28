# Phase 4: Opponent Sampling & Stability

**Goal:** Prevent "cycling" and catastrophic forgetting during self-play.

## 1. Fictitious Self-Play (FSP) (Short-Term)

**Current State:**
Naive self-play (Latest model vs Latest model).

**Proposal:**
Maintain a pool of past model checkpoints and sample opponents from this pool.

**Action Plan:**

1.  Save a model checkpoint every N steps (e.g., every 500 updates).
2.  Maintain a list/pool of paths to these checkpoints.
3.  For each training game/episode:
    - 80% probability: Play against the latest model (self-play).
    - 20% probability: Play against a randomly selected past checkpoint from the pool.
4.  Load the opponent weights into the `black_model` (or `white_model`) for that episode.

**Expected Gain:**
Prevents the agent from "forgetting" how to beat earlier strategies and stabilizes the learning curve.

## 2. League Training (Long-Term / Optional)

**Current State:**
Single agent population.

**Proposal:**
Implement a league of agents with different roles (Main, League Exploiter, Main Exploiter).

**Action Plan:**
(This is complex and likely overkill for the current stage, but worth noting for future scaling).
