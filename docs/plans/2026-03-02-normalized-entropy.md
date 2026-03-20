# Normalized Entropy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace raw entropy with normalized entropy in loss computation and add both metrics to training logs.

**Architecture:** `StepRecord` stores raw entropy + num_legal per model. `_backpropagate_for_model` normalizes at consumption. `MetricsAggregator` tracks both raw and normalized windowed averages.

**Tech Stack:** PyTorch, existing training pipeline

---

### Task 1: Add `num_legal_white/black` to `StepRecord`

**Files:**
- Modify: `src/core/types.py:19-30`
- Modify: `tests/model/test_returns.py` (`_make_step` helper + multi-env test)
- Modify: `tests/integration/test_training_desync.py` (mock envs)

**Step 1: Add fields to StepRecord**

In `src/core/types.py`, add after `entropy_black`:

```python
num_legal_white: Tensor   # [num_envs], count of legal moves for white
num_legal_black: Tensor   # [num_envs], count of legal moves for black
```

**Step 2: Update `_make_step` in test_returns.py**

Add to the helper:

```python
num_legal_white=torch.ones(n) * 20,
num_legal_black=torch.ones(n) * 20,
```

And the same for the inline `StepRecord` in `test_returns_multi_env`.

**Step 3: Update mock envs in test_training_desync.py**

Both `DesyncVectorEnv` and `AllDoneVectorEnv` `StepRecord` constructors
don't exist — they use real training. No change needed there.

**Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: FAIL — `StepRecord.__init__() missing required positional argument: 'num_legal_white'`
(from `_collect_rollout` which doesn't pass the new fields yet)

**Step 5: Update `_collect_rollout` in training.py**

In `src/model/training.py`, in the rollout loop, add tensors:

```python
nl_w = torch.ones(num_envs)
nl_b = torch.ones(num_envs)
```

Inside the per-model loop, after `ent_w[mask] = dist.entropy()`:

```python
nl_w[mask] = env_step.legal_actions_mask[mask].sum(dim=1).float()
```

And same for black:

```python
nl_b[mask] = env_step.legal_actions_mask[mask].sum(dim=1).float()
```

Add to `StepRecord(...)`:

```python
num_legal_white=nl_w,
num_legal_black=nl_b,
```

**Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS (31 tests)

**Step 7: Commit**

```bash
git add src/core/types.py src/model/training.py tests/model/test_returns.py
git commit -m "feat: add num_legal_white/black to StepRecord for entropy normalization"
```

---

### Task 2: Normalize entropy in `_backpropagate_for_model`

**Files:**
- Modify: `src/model/training.py:141-193` (`_backpropagate_for_model`)

**Step 1: Write failing test for normalized entropy**

Create `tests/model/test_normalized_entropy.py`:

```python
import torch
import math
from torch.testing import assert_close


def test_normalized_entropy_uniform_distribution():
    """Uniform distribution over N actions has normalized entropy = 1.0."""
    num_legal = 20.0
    raw_entropy = math.log(num_legal)
    log_legal = max(math.log(num_legal), 1e-8)
    normalized = raw_entropy / log_legal
    assert_close(torch.tensor(normalized), torch.tensor(1.0))


def test_normalized_entropy_forced_move_no_nan():
    """Forced move (1 legal action) must not produce inf or NaN."""
    raw_entropy = torch.tensor(0.0)
    num_legal = torch.tensor(1.0)
    log_legal = torch.clamp(torch.log(num_legal), min=1e-8)
    normalized = raw_entropy / log_legal
    assert torch.isfinite(normalized), f"Got {normalized}"
    assert_close(normalized, torch.tensor(0.0))


def test_normalized_entropy_batch():
    """Vectorized normalization across a batch."""
    raw = torch.tensor([2.0, 0.5, 0.0])
    num_legal = torch.tensor([10.0, 5.0, 1.0])
    log_legal = torch.clamp(torch.log(num_legal), min=1e-8)
    normalized = raw / log_legal
    assert torch.all(torch.isfinite(normalized))
    assert_close(normalized[0], torch.tensor(2.0 / math.log(10.0)))
    assert_close(normalized[1], torch.tensor(0.5 / math.log(5.0)))
    assert_close(normalized[2], torch.tensor(0.0))
```

**Step 2: Run test to verify it passes (pure math, no code dependency)**

Run: `.venv/bin/python -m pytest tests/model/test_normalized_entropy.py -v`
Expected: PASS (these are formula tests)

**Step 3: Add `filtered_num_legal` collection in `_backpropagate_for_model`**

In the filtering loop, after collecting `ent`:

```python
nl = (step.num_legal_white if model_id == WHITE else step.num_legal_black)[is_mine]
filtered_num_legal.append(nl)
```

Add `filtered_num_legal = []` at the top with the other lists.

**Step 4: Replace raw entropy with normalized in loss**

Replace:

```python
entropy_bonus = cat_entropies.mean()
```

With:

```python
cat_num_legal = torch.cat(filtered_num_legal)
log_legal = torch.clamp(torch.log(cat_num_legal), min=1e-8)
normalized_entropy = cat_entropies / log_legal
entropy_bonus = normalized_entropy.mean()
```

**Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/model/training.py tests/model/test_normalized_entropy.py
git commit -m "feat: normalize entropy in loss using log(num_legal_moves)"
```

---

### Task 3: Return entropy metrics from `_backpropagate_for_model`

**Files:**
- Modify: `src/model/training.py:141-193` (`_backpropagate_for_model` return)
- Modify: `src/model/training.py:222-260` (`run_chess_training` caller)

**Step 1: Change return type to dict**

In `_backpropagate_for_model`, replace both return statements:

Early return (no data):
```python
return {"total_loss": torch.tensor(0.0), "entropy_raw": 0.0, "entropy_normalized": 0.0}
```

Normal return:
```python
return {
    "total_loss": total_loss,
    "entropy_raw": cat_entropies.mean().item(),
    "entropy_normalized": normalized_entropy.mean().item(),
}
```

**Step 2: Update `run_chess_training` to use dict**

Replace:

```python
loss_w = _backpropagate_for_model(...)
loss_b = _backpropagate_for_model(...)
total_loss = loss_w.item() + loss_b.item()
```

With:

```python
result_w = _backpropagate_for_model(...)
result_b = _backpropagate_for_model(...)
total_loss = result_w["total_loss"].item() + result_b["total_loss"].item()
```

**Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/model/training.py
git commit -m "refactor: return entropy metrics dict from _backpropagate_for_model"
```

---

### Task 4: Add entropy tracking to `MetricsAggregator`

**Files:**
- Modify: `src/training/metrics.py`
- Modify: `tests/training/test_metrics_aggregation.py`
- Modify: `src/model/training.py` (wire up in `run_chess_training`)

**Step 1: Write failing test**

Add to `tests/training/test_metrics_aggregation.py`:

```python
def test_entropy_metrics_accumulate_and_average():
    m = MetricsAggregator()
    m.add_entropy(raw=2.0, normalized=0.8)
    m.add_entropy(raw=1.0, normalized=0.6)
    result = m.episode_summary()
    assert result["mean_entropy_raw"] == 1.5
    assert result["mean_entropy_normalized"] == 0.7


def test_entropy_metrics_reset_between_windows():
    m = MetricsAggregator()
    m.add_entropy(raw=2.0, normalized=0.8)
    first = m.episode_summary()
    assert first["mean_entropy_raw"] == 2.0

    m.add_entropy(raw=1.0, normalized=0.5)
    second = m.episode_summary()
    assert second["mean_entropy_raw"] == 1.0
    assert second["mean_entropy_normalized"] == 0.5


def test_entropy_metrics_zero_when_no_data():
    m = MetricsAggregator()
    result = m.episode_summary()
    assert result["mean_entropy_raw"] == 0.0
    assert result["mean_entropy_normalized"] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/training/test_metrics_aggregation.py -v`
Expected: FAIL — `add_entropy` not found

**Step 3: Implement `add_entropy` in MetricsAggregator**

In `_reset_all`, add:

```python
self._entropy_raw_sum = 0.0
self._entropy_norm_sum = 0.0
self._entropy_count = 0
```

New method:

```python
def add_entropy(self, raw: float, normalized: float):
    self._entropy_raw_sum += raw
    self._entropy_norm_sum += normalized
    self._entropy_count += 1
```

In `episode_summary()`, add to the dict:

```python
"mean_entropy_raw": self._entropy_raw_sum / max(self._entropy_count, 1),
"mean_entropy_normalized": self._entropy_norm_sum / max(self._entropy_count, 1),
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/training/test_metrics_aggregation.py -v`
Expected: ALL PASS

**Step 5: Wire up in `run_chess_training`**

After computing `result_w` and `result_b`, add:

```python
metrics.add_entropy(
    raw=(result_w["entropy_raw"] + result_b["entropy_raw"]) / 2,
    normalized=(result_w["entropy_normalized"] + result_b["entropy_normalized"]) / 2,
)
```

**Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/training/metrics.py src/model/training.py tests/training/test_metrics_aggregation.py
git commit -m "feat: track normalized and raw entropy in MetricsAggregator"
```

---

### Task 5: Update roadmap and cleanup

**Files:**
- Modify: `roadmap/01_immediate_fixes.md`
- Delete: `src/losses/entropy_regularization.py` (unused after inlining)

**Step 1: Mark roadmap item as done**

In `roadmap/01_immediate_fixes.md`, replace section 3 header with:

```markdown
## ~~3. Log Normalized Entropy~~ DONE
```

Add note: "Normalized entropy used in loss and both raw + normalized logged in MetricsAggregator."

**Step 2: Check if `entropy_regularization.py` is imported anywhere**

Run: `rg "entropy_regularization" src/`
Expected: no matches (was inlined in previous refactor)

**Step 3: Delete unused file if confirmed**

```bash
rm src/losses/entropy_regularization.py
```

**Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add roadmap/01_immediate_fixes.md
git rm src/losses/entropy_regularization.py
git commit -m "chore: mark normalized entropy done in roadmap, remove unused entropy module"
```
