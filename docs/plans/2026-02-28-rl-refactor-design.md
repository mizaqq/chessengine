# RL Refactor Design (PyTorch-first)

## Context and Goals

This design captures the approved refactor direction:

- Prioritize readability and extensibility.
- Keep the stack PyTorch + OpenSpiel for now.
- Optimize for rapid experimentation with different RL approaches.
- Keep architecture ready for future vectorized and async environments.

Primary success criteria selected:

1. Clean layered architecture with no "god functions".
2. Easy environment swapping through stable interfaces.
3. Easy model/loss swapping to test different RL strategies.

## Recommended Approach

We use a ports-and-adapters style architecture (Option B):

- Core training logic depends on interfaces, not concrete env/model details.
- OpenSpiel is wrapped behind environment adapters.
- Algorithms and losses are modular and composable.
- Training loop orchestrates data collection, optimization, and logging.

This has a higher upfront setup cost than a lightweight cleanup, but gives much better long-term speed for RL experimentation.

## Target Module Layout

```text
src/
  core/
    interfaces.py
    types.py
  envs/
    open_spiel_env.py
    open_spiel_vector_env.py
  algorithms/
    a2c/
      algorithm.py
  losses/
    policy_gradient.py
    value_mse.py
    entropy_regularization.py
    composed_loss.py
  training/
    rollout_collector.py
    trainer.py
    evaluator.py
    metrics.py
  configs/
    train_default.yaml
  entrypoints/
    train.py
    evaluate.py
```

## Core Contracts (Design-Level)

The goal is stable boundaries so internals can evolve safely.

- `Env` / `VectorEnv`
  - `reset() -> EnvStep`
  - `step(actions) -> EnvStep`
- `EnvStep`
  - `obs`, `legal_actions_mask`, `reward`, `done`, `info`
- `RolloutBatch`
  - Fixed-shape trajectory tensors + bootstrap fields.
- `Algorithm.compute_targets(batch) -> TargetBatch`
  - Returns algorithm-specific targets (returns, advantages, etc).
- `LossModule.compute(batch, targets, outputs) -> LossOutput`
  - Returns named losses and `total_loss`.

## Data Flow (One Training Update)

1. `VectorEnv` emits standardized observations and legal-action masks.
2. `RolloutCollector` gathers N steps into `RolloutBatch`.
3. `Algorithm` computes targets from rollout data.
4. `LossModule` computes `policy/value/entropy/total` losses.
5. `Trainer` performs backward/step and records metrics.

The trainer should not know OpenSpiel internals or algorithm-specific math details.

## Reliability and Observability

### Runtime Guards

- Validate tensor shapes and batch dimensions at module boundaries.
- Fail fast on NaN/Inf in losses and invalid action masks.
- Use one terminal-state accounting path to avoid double-counting.

### Metrics

- Central `MetricsAggregator` per update/episode.
- Log normalized entropy (aligned with optimized objective).
- Keep separate metric groups:
  - data collection
  - optimization
  - game outcomes (wins/losses/draws)

### Debug Mode

- Deterministic seed mode for reproducibility.
- Optional env-level trace for step-by-step diagnostics.

## Migration Strategy (Incremental, No Big-Bang Rewrite)

### Step 0: Baseline

- Run short baseline training jobs and store metrics snapshot.
- Use this snapshot as regression reference.

### Step 1: Contracts and Types

- Add `core/interfaces.py` and `core/types.py`.
- Add contract tests for invariants and shapes.

### Step 2: Environment Adapters

- Extract OpenSpiel logic into `envs/`.
- Add reset/step/done/mask consistency tests.

### Step 3: Rollout + Metrics Split

- Move collection and aggregation out of monolithic training function.
- Fix immediate issues:
  - win/draw double-counting
  - episode metric accumulation
  - normalized entropy logging

### Step 4: Modular Algorithm + Loss

- Implement A2C in new API as first algorithm module.
- Use `ComposedLoss` for swap-friendly loss design.

### Step 5: Vectorization Path

- Add sync `VectorEnv` first.
- Add async/multiprocessing behind same interface later.

## Testing Strategy

- Unit tests for loss components, returns/advantage math, metrics aggregation.
- Contract tests for env adapters and rollout invariants.
- Integration smoke tests (short runs) for:
  - no NaN/Inf
  - stable loop execution
  - no terminal-state double counting
- Regression checks against baseline metrics with tolerance windows.

## Out of Scope for This Phase

- JAX/PGX migration.
- Full framework registry system.
- Distributed training across machines.

## Risks and Mitigations

- Risk: Upfront complexity and more files.
  - Mitigation: strict interface boundaries and incremental migration steps.
- Risk: Hidden behavior regressions during extraction.
  - Mitigation: baseline snapshot + smoke/regression checks at each step.
- Risk: Overengineering too early.
  - Mitigation: keep first algorithm limited to A2C and add features only when needed.
