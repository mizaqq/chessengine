# Training Loop Refactor — Design Document

## Context

The project has two new vectorized environments (`OpenSpielVectorEnv`, `OpenSpielAsyncVectorEnv`) using a unified `EnvStep` API, but the training loop (`src/model/training.py`) still depends on the legacy `EnvSpawner` with its pull-based interface (`get_current_states()`, `move()`, `get_done()`, etc.). This refactor rewrites the training loop to use `EnvStep` directly and removes the legacy environment.

## Approach: Big Bang Rewrite

Rewrite `training.py` from scratch in CleanRL style, directly targeting the `EnvStep` API. Remove legacy `EnvSpawner` simultaneously. Chosen over adapter-bridge (throwaway code for already-messy training loop) and parallel implementation (unnecessary duplication for simple A2C logic).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Loop style | CleanRL inline (no `model_acc()`) | More readable, standard in RL community |
| VectorBuffer | Decouple, keep as standalone module | May need for future off-policy RL |
| Terminal rewards | Configurable via YAML | Hyperparameter tuning without code changes |
| Player handling | Player-agnostic loop | Handles auto-reset desync naturally |
| Current player info | `current_player: Tensor` field in `EnvStep` | Type-safe, clean API |

## Section 1: EnvStep Extension

Add `current_player: Tensor` to `EnvStep` dataclass (`src/core/types.py`):

```python
@dataclass
class EnvStep:
    obs: Tensor
    legal_actions_mask: Tensor
    reward: Tensor
    done: Tensor
    current_player: Tensor  # [num_envs], 0=white, 1=black
    info: Dict[str, Any]
```

Both `OpenSpielVectorEnv` and `OpenSpielAsyncVectorEnv` populate this field. After auto-reset, `current_player=0` (white starts).

## Section 2: Training Loop Architecture

### Player-Agnostic Inner Loop

Each step queries the model matching `current_player` per env. The batch is split by player for forward passes, then merged:

```python
env_step = envs.reset()
for step in range(N):
    white_mask = (env_step.current_player == 0)
    black_mask = (env_step.current_player == 1)

    # Forward through respective models, sample actions
    ...

    env_step = envs.step(actions)

    # Collect StepRecord with player, value, log_prob, entropy, reward, done, terminal_rewards
```

### Returns Computation with Cross-Player Done Propagation

When a game ends on the opponent's step, it must cut the return chain for the current model. The `effective_done` mechanism handles this via pure tensor operations:

```python
R = bootstrap_value  # [num_envs]
for i in reversed(range(N)):
    step = steps[i]
    is_mine = (step.player == model_id)

    # A: cut chain on done + inject terminal reward
    R = R * (1 - step.done.float()) + step.terminal_r[model_id]

    # B: update R only for this model's steps
    new_R = step.reward_for[model_id] + gamma * R
    R = torch.where(is_mine, new_R, R)
```

This is N iterations of 4 tensor operations on [num_envs] vectors (~0.001ms total). Terminal rewards are precomputed as tensors during collection to avoid Python dict iteration during returns.

### Why Cross-Player Done Is Necessary

Without it, returns from a finished game leak into a new game after auto-reset. Example: if black checkmates at step 3, white's return at step 2 would incorrectly include rewards from a new game starting at step 4. The `done` at step 3 cuts white's chain even though white didn't act at step 3.

### Backpropagation

Per-model: filter steps where `is_mine`, compute advantages, aggregate policy/value/entropy losses using existing `ComposedLoss`. Same A2C algorithm, different data source.

## Section 3: Entrypoint Changes

`src/entrypoints/train.py` and `src/main.py`:
- Replace `EnvSpawner` with env factory based on config `env_type` ("async" | "sync")
- Pass `terminal_rewards` config to `run_chess_training()`
- YAML config gets new keys: `terminal_rewards`, `env_type`

## Section 4: Cleanup

### Remove
- `src/environment/environment.py` (after extracting `VectorBuffer`)
- `src/core/interfaces.py` (empty file)
- `model_acc()`, `check_win()` from training.py
- `piece_difference_from_tensor()` from `src/utils/utils.py` (if unused elsewhere)

### Move
- `VectorBuffer` → `src/training/vector_buffer.py` (standalone, decoupled)

### Modify
- `src/core/types.py` — add `current_player`
- `src/envs/open_spiel_vector_env.py` — populate `current_player`
- `src/envs/open_spiel_async_vector_env.py` — worker sends `current_player`
- `src/model/training.py` — full rewrite
- `src/entrypoints/train.py` — new env + config
- `src/main.py` — same

### Keep unchanged
- `src/envs/open_spiel_env.py`
- `src/losses/` (all)
- `src/model/chess_model.py`
- `src/training/metrics.py`
