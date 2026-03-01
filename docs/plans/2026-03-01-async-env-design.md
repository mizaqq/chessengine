# Async Environment Design Document

## Overview
To speed up data collection for reinforcement learning, we will implement vectorized environments. This allows multiple game instances to run in parallel, batching their observations for efficient neural network inference.

## Goals
1.  **Increase Throughput:** Achieve 4x-8x speedup in data collection using CPU multiprocessing.
2.  **Future-Proofing:** Design a `VectorEnv` interface that supports future GPU-native implementations (JAX/Pgx) without changing the training loop.
3.  **Compatibility:** Reuse existing `OpenSpielEnv` logic.

## Architecture

### 1. `VectorEnv` Interface
An abstract base class defining the contract for all vectorized environments.

```python
class VectorEnv(ABC):
    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, Any]: ...
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]: ...
    @abstractmethod
    def close(self): ...
```

### 2. `AsyncVectorEnv` Implementation
A concrete implementation of `VectorEnv` using `torch.multiprocessing`.

*   **Master Process:**
    *   Manages a list of `Pipe` connections to worker processes.
    *   Sends actions to workers.
    *   Receives observations, rewards, and done flags.
    *   Stacks results into PyTorch tensors.
*   **Worker Processes:**
    *   Each worker runs an instance of `OpenSpielEnv`.
    *   Waits for commands (`step`, `reset`, `close`) via `Pipe`.
    *   Executes the command and sends back the result.
    *   Handles auto-resetting when an episode finishes.

### 3. Data Flow
1.  **Agent** generates a batch of actions `[A1, A2, ..., An]`.
2.  **AsyncVectorEnv** splits actions and sends `Ai` to Worker `i`.
3.  **Workers** execute `env.step(Ai)` in parallel.
4.  **Workers** return `(State_i, Reward_i, Done_i, Info_i)`.
    *   If `Done_i` is True, `State_i` is the *reset* state ($S_0$), and `Info_i` contains the terminal state ($S_T$).
5.  **AsyncVectorEnv** aggregates results into `BatchState`, `BatchReward`, `BatchDone`, `BatchInfo`.
6.  **Agent** uses `BatchState` for the next inference.

## Key Components

*   **`src/core/interfaces.py`**: Defines `VectorEnv`.
*   **`src/environment/async_vector_env.py`**: Implements `AsyncVectorEnv` and the worker function.
*   **`src/envs/open_spiel_env.py`**: Existing single environment wrapper (unchanged).

## Trade-offs
*   **Pros:** Immediate speedup, reuses existing logic, robust.
*   **Cons:** IPC overhead (pickling), CPU bottleneck compared to GPU-native.

## TODOs

### Integration with Training Loop
- [ ] **Add `info["game_result"]` to worker** — when `done=True`, worker should include the game outcome (`white_win`, `black_win`, `draw`) in `info`. Training loop needs this because it can no longer access `env.env.get_time_step().rewards` directly (env lives in another process).
- [ ] **Adapt `training.py` to use `AsyncVectorEnv`** — replace `EnvSpawner` with `AsyncVectorEnv`. The training loop currently calls `envs.get_current_states()`, `envs.get_current_actions()`, `envs.get_done()` etc. — these need to be replaced with the `step()`/`reset()` API. Terminal reward logic (+2/-2/-0.5) stays in the training loop but reads from `info["game_result"]`.
- [ ] **Handle white/black alternation** — current `EnvSpawner.one_iteration()` handles white move + random black move in one call. `AsyncVectorEnv.step()` advances one player per call. Training loop needs to be adapted for this.

### Performance
- [ ] **Profile IPC overhead** — measure time spent in `Pipe.send()`/`recv()` vs actual `env.step()`. If IPC > 30% of total step time, consider shared memory approach.
- [ ] **Benchmark straggler effect** — measure variance of per-worker step times. If high, consider double-buffering.
- [ ] **Consider shared memory** — if profiling shows IPC is the bottleneck, replace Pipe with `multiprocessing.shared_memory` for observations (5KB per env is small, but at 64+ envs it adds up).

### Future Architecture
- [ ] **Implement `JaxVectorEnv`** — adhering to the same `VectorEnv` interface for massive parallelism on GPU (100-1000x speedup). Investigate Pgx library.
- [ ] **True async collection** — current `step()` is synchronous (waits for all workers). For large worker counts, implement double-buffering or IMPALA-style async to eliminate straggler problem.
