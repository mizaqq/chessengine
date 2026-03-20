# Phase 1: Training Infrastructure & Environment Scalability

**Goal:** Increase environment throughput to speed up data collection.

## ~~1. Vectorized Environments~~ DONE

Implemented two vectorized environment wrappers with unified `EnvStep` API:

- **`OpenSpielVectorEnv`** (sync) — sequential stepping, simplest implementation
- **`OpenSpielAsyncVectorEnv`** (async) — `torch.multiprocessing` with Pipes, `fork` context

Both support auto-reset, `current_player` tracking, piece-difference rewards, and
terminal game result reporting via `info["game_results"]`. Configurable via `env_type`
in YAML config. Benchmark: async is **1.14x faster** than sync with 12 envs.

Legacy `EnvSpawner` removed. `VectorBuffer` moved to standalone module.

## 2. JAX / GPU-Native Environments (Long-Term)

**Current State:**
CPU-based OpenSpiel environment. Data transfer between CPU (env) and GPU (model) is a bottleneck.

**Proposal:**
Port the environment logic to JAX using libraries like **Pgx**.

**Action Plan:**

1.  Investigate **Pgx** (JAX-based board game simulators).
2.  Rewrite the environment step logic to run entirely on the GPU.
3.  This eliminates CPU-GPU data transfer and allows for massive parallelism (thousands of environments).

**Expected Gain:**
100x - 1000x speedup.
