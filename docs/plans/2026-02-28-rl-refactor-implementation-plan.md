# RL Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the training stack into a modular, testable architecture where environment adapters, RL algorithms, and loss functions are easy to swap while staying on PyTorch + OpenSpiel.

**Architecture:** Use ports-and-adapters boundaries (`core/interfaces.py`, `core/types.py`) so trainer orchestration is decoupled from OpenSpiel and algorithm math details. Move from monolithic training logic to composed modules (`envs`, `training`, `algorithms`, `losses`) with contract tests and smoke tests to prevent regressions.

**Tech Stack:** Python, PyTorch, OpenSpiel, pytest

---

### Task 1: Create Core Contracts and Type Models

**Files:**
- Create: `src/core/interfaces.py`
- Create: `src/core/types.py`
- Create: `tests/core/test_interfaces_contracts.py`

**Step 1: Write the failing test**

```python
from dataclasses import fields
from src.core.types import EnvStep, RolloutBatch


def test_env_step_has_required_fields():
    names = {f.name for f in fields(EnvStep)}
    assert {"obs", "legal_actions_mask", "reward", "done", "info"} <= names


def test_rollout_batch_has_minimum_training_fields():
    names = {f.name for f in fields(RolloutBatch)}
    assert {"obs", "actions", "rewards", "dones", "legal_actions_mask"} <= names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_interfaces_contracts.py -v`  
Expected: FAIL (`ModuleNotFoundError` or missing symbols)

**Step 3: Write minimal implementation**

```python
# src/core/types.py
from dataclasses import dataclass
from typing import Any, Dict
import torch

Tensor = torch.Tensor


@dataclass
class EnvStep:
    obs: Tensor
    legal_actions_mask: Tensor
    reward: Tensor
    done: Tensor
    info: Dict[str, Any]


@dataclass
class RolloutBatch:
    obs: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    legal_actions_mask: Tensor
```

```python
# src/core/interfaces.py
from abc import ABC, abstractmethod
from src.core.types import EnvStep


class VectorEnv(ABC):
    @abstractmethod
    def reset(self) -> EnvStep:
        ...

    @abstractmethod
    def step(self, actions):
        ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_interfaces_contracts.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/interfaces.py src/core/types.py tests/core/test_interfaces_contracts.py
git commit -m "refactor: add core RL contracts and batch types"
```

---

### Task 2: Extract OpenSpiel Adapter Layer

**Files:**
- Create: `src/envs/open_spiel_env.py`
- Create: `src/envs/open_spiel_vector_env.py`
- Create: `tests/envs/test_open_spiel_vector_env.py`
- Modify: `src/environment/environment.py`

**Step 1: Write the failing test**

```python
import torch
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv


def test_vector_env_reset_returns_valid_shapes():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    assert step.obs.shape[0] == 2
    assert step.legal_actions_mask.shape == (2, 4674)
    assert step.done.dtype == torch.bool
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/envs/test_open_spiel_vector_env.py::test_vector_env_reset_returns_valid_shapes -v`  
Expected: FAIL (class missing)

**Step 3: Write minimal implementation**

```python
# Implement OpenSpielVectorEnv with reset() and step() returning EnvStep.
# Keep current behavior compatibility by wrapping existing WrappedEnv logic.
# In src/environment/environment.py, keep compatibility shims that delegate to new module.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/envs/test_open_spiel_vector_env.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/envs/open_spiel_env.py src/envs/open_spiel_vector_env.py tests/envs/test_open_spiel_vector_env.py src/environment/environment.py
git commit -m "refactor: move OpenSpiel logic behind vector env adapter"
```

---

### Task 3: Split Rollout Collection and Metrics Aggregation

**Files:**
- Create: `src/training/rollout_collector.py`
- Create: `src/training/metrics.py`
- Create: `tests/training/test_metrics_aggregation.py`
- Modify: `src/model/training.py`

**Step 1: Write the failing test**

```python
from src.training.metrics import MetricsAggregator


def test_episode_counters_accumulate_over_multiple_steps():
    m = MetricsAggregator()
    m.add_step(empty_masks=2, illegal_samples=1)
    m.add_step(empty_masks=1, illegal_samples=3)
    result = m.episode_summary()
    assert result["episode_empty_masks"] == 3
    assert result["episode_illegal_samples"] == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_metrics_aggregation.py -v`  
Expected: FAIL (class missing)

**Step 3: Write minimal implementation**

```python
# Add MetricsAggregator with per-step accumulation and reset hooks.
# Move terminal-result accounting to a single path.
# Update training logs to include normalized entropy metric.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_metrics_aggregation.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/metrics.py src/training/rollout_collector.py src/model/training.py tests/training/test_metrics_aggregation.py
git commit -m "refactor: separate rollout collection and reliable metrics aggregation"
```

---

### Task 4: Introduce Modular Loss Composition

**Files:**
- Create: `src/losses/policy_gradient.py`
- Create: `src/losses/value_mse.py`
- Create: `src/losses/entropy_regularization.py`
- Create: `src/losses/composed_loss.py`
- Create: `tests/losses/test_composed_loss.py`
- Modify: `src/model/training.py`

**Step 1: Write the failing test**

```python
import torch
from src.losses.composed_loss import ComposedLoss


def test_composed_loss_returns_named_terms_and_total():
    loss = ComposedLoss()
    out = loss.compute(
        policy_loss=torch.tensor(1.0),
        value_loss=torch.tensor(2.0),
        entropy_bonus=torch.tensor(0.5),
        entropy_coef=0.1,
    )
    assert set(out.keys()) == {"policy_loss", "value_loss", "entropy_bonus", "total_loss"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/losses/test_composed_loss.py -v`  
Expected: FAIL (module missing)

**Step 3: Write minimal implementation**

```python
# Implement small pure functions/classes per loss type.
# Compose total loss in composed_loss.py and return named components.
# Replace inlined loss math in training.py with composed module calls.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/losses/test_composed_loss.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/losses/*.py tests/losses/test_composed_loss.py src/model/training.py
git commit -m "refactor: compose RL losses into swappable modules"
```

---

### Task 5: Entry Point and Config Wiring

**Files:**
- Create: `src/configs/train_default.yaml`
- Create: `src/entrypoints/train.py`
- Modify: `src/main.py`
- Create: `tests/integration/test_training_smoke.py`

**Step 1: Write the failing test**

```python
def test_training_smoke_runs_short_job_without_nan():
    # Run 1-2 short updates in test mode and assert no NaN metrics.
    # This test protects against orchestration regressions.
    assert True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_training_smoke.py -v`  
Expected: FAIL once replaced with real smoke assertion

**Step 3: Write minimal implementation**

```python
# Add config-driven train entrypoint.
# Keep src/main.py as compatibility shim calling new entrypoint.
# Wire trainer/env/algorithm/loss from config.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_training_smoke.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/configs/train_default.yaml src/entrypoints/train.py src/main.py tests/integration/test_training_smoke.py
git commit -m "refactor: add config-driven training entrypoint and smoke test"
```

---

### Task 6: Regression Verification and Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/plans/2026-02-28-rl-refactor-migration-notes.md`

**Step 1: Run short baseline and refactor smoke runs**

Run:
- `python -m src.entrypoints.train --config src/configs/train_default.yaml --max-updates 5 --seed 42`
- Collect key metrics (`loss`, `normalized_entropy`, `wins/draws/losses` counters)

Expected: Runs complete without crashes or NaN/Inf.

**Step 2: Compare with baseline tolerance window**

Document differences and acceptable tolerance in migration notes.

**Step 3: Update README usage**

Document new entrypoint and module layout.

**Step 4: Final verification**

Run: `pytest -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-02-28-rl-refactor-migration-notes.md
git commit -m "docs: document refactor architecture and migration checks"
```
