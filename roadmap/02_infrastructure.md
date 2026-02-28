# Phase 1: Training Infrastructure & Environment Scalability

**Goal:** Increase environment throughput to speed up data collection.

## 1. Vectorized Environments (Immediate Priority)

**Current State:**
`EnvSpawner` iterates through environments sequentially in a Python loop.

**Proposal:**
Implement a vectorized environment wrapper using `torch.multiprocessing` or adapt `gym.vector.AsyncVectorEnv` for OpenSpiel.

**Action Plan:**

1.  Create a `VectorizedChessEnv` class that manages multiple OpenSpiel instances.
2.  Use `multiprocessing` to step environments in parallel subprocesses.
3.  Ensure the main process receives a batch of observations (states) and returns a batch of actions.
4.  This allows the GPU to process a batch of states while the CPUs are stepping the environments for the next turn.

**Expected Gain:**
4x - 8x speedup in data collection.

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
