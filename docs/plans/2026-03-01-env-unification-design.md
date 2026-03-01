# Environment Unification Design

## Goal
Unify sync and async vectorized environments to share the same API (`EnvStep`), both with auto-reset and game result reporting. Remove the `VectorEnv` ABC.

## Changes
1. Remove `VectorEnv` ABC from `src/core/interfaces.py`
2. Move async env to `src/envs/open_spiel_async_vector_env.py`
3. Add auto-reset + `info["terminal_observation"]` + `info["game_result"]` to both envs
4. Both return `EnvStep` from `src/core/types.py`
5. Delete old `src/environment/async_vector_env.py`
6. Update tests

## Unchanged
- `src/envs/open_spiel_env.py` — single env wrapper
- `src/environment/environment.py` — legacy EnvSpawner
- `src/core/types.py` — EnvStep, RolloutBatch
