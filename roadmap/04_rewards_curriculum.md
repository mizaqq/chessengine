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

## 2. Endgame Curriculum (Mid-Term) — evidence 2026-09-15: models miss 96% of mates-in-one; see roadmap/06 §0

**Status 2026-09-16:** first step implemented as change `mate-in-one-curriculum`:
with probability `puzzle_fraction` (0.5) a game starts from a Lichess mate-in-one
puzzle (reverse curriculum by start state, Florensa et al. 2017; Salimans & Chen
2018). Held-out puzzle metrics in `src/eval/puzzles.py`. Results in
`experiments/mate-in-one-curriculum/`. Next widening: `mateIn2`, then an adaptive
fraction. The generated-endgame plan below remains the alternative source of
positions.

**Current State (before 2026-09-16):**
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

## ~~3. Outcome Modeling~~ PARTIALLY DONE

Terminal rewards (win/loss/draw) are now configurable per-player via YAML:

```yaml
terminal_rewards:
  win: 2.0
  loss: -2.0
  draw: -0.5
```

Rewards are precomputed as vectorized tensors (`terminal_r_white`, `terminal_r_black`)
and injected into the returns computation. Draw is currently -0.5 (small penalty).

**Remaining:**
- 3-head value network (Win/Loss/Draw probabilities) — not yet implemented
- Asymmetric draw rewards (different for white vs black) — not yet implemented
