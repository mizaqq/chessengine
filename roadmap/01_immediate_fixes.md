# Phase 0: Immediate Codebase Fixes

These are critical bugs and improvements needed in `src/model/training.py` before any major architectural changes.

## ~~1. Fix Win/Draw Double-Counting~~ DONE

Resolved by full rewrite of training loop. Legacy `check_win()` and `EnvSpawner` removed.
Game results now handled via `info["game_results"]` from auto-resetting vector envs,
processed once in `_collect_rollout`.

## ~~2. Fix Metric Accumulation~~ DONE

`MetricsAggregator` now properly accumulates metrics with windowed summaries.
`episode_summary()` returns stats since last call (including win rates) and auto-resets.

## ~~3. Log Normalized Entropy~~ DONE

Entropy normalization implemented at the point of consumption:
- `_backpropagate_for_model` normalizes entropy for loss: `raw / clamp(log(num_legal), 1e-8)`
- `MetricsAggregator` tracks both `mean_entropy_raw` and `mean_entropy_normalized`
- Forced moves (1 legal action) handled safely via clamping
- Step-count weighted averaging prevents bias when one model has no data
- Unused `compute_entropy_bonus()` removed from `src/losses/`
