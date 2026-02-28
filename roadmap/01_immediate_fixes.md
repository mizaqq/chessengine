# Phase 0: Immediate Codebase Fixes

These are critical bugs and improvements needed in `src/model/training.py` before any major architectural changes.

## 1. Fix Win/Draw Double-Counting

**Issue:**
The `check_win()` function is currently called in two places:

1. Inside the rollout loop when `any(envs.get_done())` is triggered.
2. Again at the start of the next episode loop before `envs.reset_all()`.

**Consequence:**
The same game result is logged twice, inflating win/loss statistics.

**Action:**

- Remove the redundant call to `check_win()` at the start of the episode loop.
- Ensure it is only called once when a terminal state is detected.

## 2. Fix Metric Accumulation

**Issue:**
`episode_empty_masks` and `episode_illegal_samples` are currently reset or overwritten in every call to `model_acc`.

**Consequence:**
These metrics only reflect the last step of an episode, rather than the total count of empty masks or illegal samples encountered throughout the entire episode.

**Action:**

- Initialize these counters at the start of the episode.
- Accumulate values across all steps in the episode.
- Log the total sum at the end of the episode.

## 3. Log Normalized Entropy

**Issue:**
The loss function now optimizes _normalized_ entropy (`entropy / log(legal_moves)`), but the logs still track raw entropy (`entropies.mean()`).

**Consequence:**
The logged entropy metric is misleading and does not reflect what the agent is actually optimizing. A decrease in raw entropy might just mean fewer legal moves available, not necessarily less exploration relative to the available options.

**Action:**

- Update the logging logic (both `pbar` and `logs.json`) to track the normalized entropy value used in the loss calculation.
