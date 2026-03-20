# Normalized Entropy in Loss and Logs

## Problem

Raw entropy (`-Σ pᵢ log pᵢ`) depends on the number of legal moves. A position
with 30 legal moves naturally has higher entropy than one with 5. Tracking or
optimizing raw entropy is misleading — a drop may just mean fewer legal moves,
not less exploration.

## Decision

Introduce **normalized entropy** = `raw_entropy / log(num_legal_moves)`, scaled
to [0, 1] where 1 = uniform over legal moves, 0 = deterministic.

- Use normalized entropy in the **loss function** (changes training behavior)
- Log **both** raw and normalized entropy in `MetricsAggregator`
- Handle forced moves (`num_legal = 1`, `log(1) = 0`) via `clamp(min=epsilon)`

## Approach: Normalization at Consumption (Approach B)

`StepRecord` stores raw entropy + `num_legal_moves` per model. Normalization
happens in `_backpropagate_for_model` (for loss) and in metrics (for logging).
This preserves raw entropy for diagnostics.

## Data Changes

`StepRecord` gets two new fields:
- `num_legal_white: Tensor`  — `[num_envs]`, legal move count per env for white
- `num_legal_black: Tensor`  — `[num_envs]`, legal move count per env for black

Populated in `_collect_rollout` from `legal_actions_mask.sum(dim=1)` at the
moment of each model's forward pass. Default = 1.0 for envs where the model
is not active (safe for log).

## Loss Normalization

In `_backpropagate_for_model`, after concatenating filtered tensors:

```python
log_legal = torch.clamp(torch.log(cat_num_legal), min=1e-8)
normalized_entropy = cat_entropies / log_legal
entropy_bonus = normalized_entropy.mean()
```

Gradient flows through `cat_entropies` (model output). `log_legal` has no grad
(derived from `legal_actions_mask`). Effect: stronger exploration push when
fewer legal moves are available.

## Return Value Change

`_backpropagate_for_model` returns a dict instead of a scalar:

```python
{"total_loss": Tensor, "entropy_raw": float, "entropy_normalized": float}
```

`run_chess_training` extracts loss for accumulation and feeds entropy values
to `MetricsAggregator`.

## Metrics

`MetricsAggregator` gets `add_entropy(raw, normalized)`. `episode_summary()`
includes `mean_entropy_raw` and `mean_entropy_normalized` (windowed averages).

## Tests

1. Update `StepRecord` in `_make_step` helper and mock envs (new `num_legal_*`)
2. New unit test: normalization math, forced move (no inf/nan), uniform = 1.0
3. Test `MetricsAggregator.add_entropy()` accumulation and windowed averages
4. Existing desync/smoke tests updated for new `StepRecord` fields
