# Phase 0: Immediate Codebase Fixes

These are critical bugs and improvements needed in `src/model/training.py` before any major architectural changes.

## ~~1. Fix Win/Draw Double-Counting~~ DONE

Resolved by full rewrite of training loop. Legacy `check_win()` and `EnvSpawner` removed.
Game results now handled via `info["game_results"]` from auto-resetting vector envs,
processed once in `_collect_rollout`.

## ~~2. Fix Metric Accumulation~~ DONE

`MetricsAggregator` now properly accumulates metrics with windowed summaries.
`episode_summary()` returns stats since last call (including win rates) and auto-resets.

## 3. Log Normalized Entropy

**Issue:**
The loss function now optimizes _normalized_ entropy (`entropy / log(legal_moves)`), but the logs still track raw entropy (`entropies.mean()`).

**Consequence:**
The logged entropy metric is misleading and does not reflect what the agent is actually optimizing. A decrease in raw entropy might just mean fewer legal moves available, not necessarily less exploration relative to the available options.

**Action:**

- Update the logging logic (both `pbar` and `logs.json`) to track the normalized entropy value used in the loss calculation.
