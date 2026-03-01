# Environment Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify sync and async vectorized environments to share the same API (`EnvStep`), both with auto-reset and game result reporting. Remove `VectorEnv` ABC.

**Architecture:** Both `OpenSpielVectorEnv` (sync) and `OpenSpielAsyncVectorEnv` (async) live in `src/envs/`, return `EnvStep`, handle auto-reset internally, and report game results via `info`. No abstract base class — duck typing.

**Tech Stack:** Python, PyTorch, torch.multiprocessing, OpenSpiel.

---

### Task 1: Add `game_result` helper to `OpenSpielEnv`

**Files:**
- Modify: `src/envs/open_spiel_env.py`
- Create: `tests/envs/test_open_spiel_env.py`

Both vector envs need to query game results. Instead of duplicating `env.env.get_time_step().rewards` logic, add a method to the single env.

**Step 1: Write the failing test**

```python
from src.envs.open_spiel_env import OpenSpielEnv


def test_game_result_returns_none_when_not_done():
    env = OpenSpielEnv()
    env.reset()
    assert env.game_result() is None


def test_game_result_returns_string_when_done():
    env = OpenSpielEnv()
    env.reset()
    # Play random moves until game ends
    import random
    while not env.is_done():
        legal = env.get_legal_actions().nonzero().tolist()
        env.step([random.choice(legal)])
    result = env.game_result()
    assert result in ("white_win", "black_win", "draw")
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/envs/test_open_spiel_env.py -v`
Expected: FAIL with `AttributeError: 'OpenSpielEnv' object has no attribute 'game_result'`

**Step 3: Write minimal implementation**

Add to `src/envs/open_spiel_env.py`:
```python
    def game_result(self):
        """Return game result string or None if game is not over."""
        if not self.is_done():
            return None
        rewards = self.env.get_time_step().rewards
        if rewards[0] > 0:
            return "white_win"
        elif rewards[1] > 0:
            return "black_win"
        return "draw"
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/envs/test_open_spiel_env.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/envs/open_spiel_env.py tests/envs/test_open_spiel_env.py
git commit -m "feat: add game_result() to OpenSpielEnv"
```

### Task 2: Upgrade `OpenSpielVectorEnv` with auto-reset, rewards, info

**Files:**
- Modify: `src/envs/open_spiel_vector_env.py`
- Modify: `tests/envs/test_open_spiel_vector_env.py`

**Step 1: Write the failing test**

```python
import torch
import random
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv


def test_vector_env_reset_returns_valid_shapes():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    assert step.obs.shape == (2, 20, 8, 8)
    assert step.legal_actions_mask.shape == (2, 4674)
    assert step.done.dtype == torch.bool


def test_vector_env_step_returns_env_step():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    legal = step.legal_actions_mask
    actions = []
    for i in range(2):
        legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
        actions.append(legal_indices[0].item())
    step = env.step(torch.tensor(actions))
    assert step.obs.shape == (2, 20, 8, 8)
    assert step.reward.shape == (2,)
    assert step.done.shape == (2,)
    assert isinstance(step.info, dict)


def test_vector_env_auto_reset():
    """After a game ends, env should auto-reset and report terminal info."""
    env = OpenSpielVectorEnv(num_envs=1)
    env.reset()
    for _ in range(500):
        step_result = env.reset() if _ == 0 else step_result
        legal = step_result.legal_actions_mask
        legal_indices = (legal[0] == 1).nonzero(as_tuple=True)[0]
        action = legal_indices[random.randint(0, len(legal_indices) - 1)].item()
        step_result = env.step(torch.tensor([action]))
        if step_result.done[0]:
            assert "terminal_observations" in step_result.info
            assert "game_results" in step_result.info
            assert step_result.info["game_results"][0] in ("white_win", "black_win", "draw")
            break
    else:
        raise AssertionError("Game never ended in 500 steps")
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/envs/test_open_spiel_vector_env.py -v`
Expected: FAIL (no auto-reset, no info keys)

**Step 3: Write minimal implementation**

Rewrite `src/envs/open_spiel_vector_env.py`:
```python
import torch
import numpy as np
from src.core.types import EnvStep
from src.envs.open_spiel_env import OpenSpielEnv

PIECE_VALUES = np.array([0, 9, 5, 3, 3, 1], dtype=np.float32)


def _piece_difference_np(state: np.ndarray) -> float:
    piece_counts = state[:12].sum(axis=(1, 2))
    white_scores = (piece_counts[0::2] * PIECE_VALUES).sum()
    black_scores = (piece_counts[1::2] * PIECE_VALUES).sum()
    return float(white_scores - black_scores)


class OpenSpielVectorEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [OpenSpielEnv() for _ in range(num_envs)]
        self._previous_states = [None] * num_envs

    def reset(self) -> EnvStep:
        for env in self.envs:
            env.reset()
        self._previous_states = [env.state() for env in self.envs]

        obs = torch.tensor(np.stack(self._previous_states)).float()
        legal = torch.stack([env.get_legal_actions() for env in self.envs])
        reward = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return EnvStep(obs=obs, legal_actions_mask=legal, reward=reward, done=done, info={})

    def step(self, actions) -> EnvStep:
        if isinstance(actions, torch.Tensor):
            actions = actions.tolist()

        terminal_observations = {}
        game_results = {}
        rewards = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if not env.is_done():
                env.step([action])

            obs = env.state()
            done = env.is_done()

            reward = _piece_difference_np(obs) - _piece_difference_np(self._previous_states[i])
            rewards.append(reward)

            if done:
                terminal_observations[i] = obs.copy()
                game_results[i] = env.game_result()
                env.reset()
                obs = env.state()

            self._previous_states[i] = obs

        all_obs = torch.tensor(np.stack(self._previous_states)).float()
        legal = torch.stack([env.get_legal_actions() for env in self.envs])
        reward_t = torch.tensor(rewards).float()
        done_t = torch.tensor([env.is_done() for env in self.envs], dtype=torch.bool)
        # Note: done_t is False after auto-reset. Use terminal_observations to detect episodes that ended.

        info = {}
        if terminal_observations:
            info["terminal_observations"] = terminal_observations
            info["game_results"] = game_results

        return EnvStep(obs=all_obs, legal_actions_mask=legal, reward=reward_t, done=done_t, info=info)
```

Note: `done_t` will be `False` after auto-reset (env already reset). The `info["terminal_observations"]` dict (keyed by env index) signals which envs finished.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/envs/test_open_spiel_vector_env.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/envs/open_spiel_vector_env.py tests/envs/test_open_spiel_vector_env.py
git commit -m "feat: upgrade OpenSpielVectorEnv with auto-reset and game results"
```

### Task 3: Create `OpenSpielAsyncVectorEnv` using `EnvStep`

**Files:**
- Create: `src/envs/open_spiel_async_vector_env.py`
- Create: `tests/envs/test_open_spiel_async_vector_env.py`

**Step 1: Write the failing test**

```python
import torch
import random
from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv


def test_async_env_reset():
    env = OpenSpielAsyncVectorEnv(num_envs=2)
    step = env.reset()
    assert step.obs.shape == (2, 20, 8, 8)
    assert step.legal_actions_mask.shape == (2, 4674)
    env.close()


def test_async_env_step():
    env = OpenSpielAsyncVectorEnv(num_envs=2)
    step = env.reset()
    actions = []
    for i in range(2):
        legal_indices = (step.legal_actions_mask[i] == 1).nonzero(as_tuple=True)[0]
        actions.append(legal_indices[0].item())
    step = env.step(torch.tensor(actions))
    assert step.obs.shape == (2, 20, 8, 8)
    assert step.reward.shape == (2,)
    assert step.done.shape == (2,)
    env.close()


def test_async_env_auto_reset_with_game_result():
    env = OpenSpielAsyncVectorEnv(num_envs=1)
    step = env.reset()
    for _ in range(500):
        legal = step.legal_actions_mask
        legal_indices = (legal[0] == 1).nonzero(as_tuple=True)[0]
        action = legal_indices[random.randint(0, len(legal_indices) - 1)].item()
        step = env.step(torch.tensor([action]))
        if "game_results" in step.info:
            assert step.info["game_results"][0] in ("white_win", "black_win", "draw")
            assert 0 in step.info["terminal_observations"]
            break
    else:
        raise AssertionError("Game never ended in 500 steps")
    env.close()
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/envs/test_open_spiel_async_vector_env.py -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

Create `src/envs/open_spiel_async_vector_env.py`:
```python
import torch.multiprocessing as mp
import torch
import numpy as np
from src.core.types import EnvStep
from src.envs.open_spiel_env import OpenSpielEnv

PIECE_VALUES = np.array([0, 9, 5, 3, 3, 1], dtype=np.float32)


def _piece_difference_np(state: np.ndarray) -> float:
    piece_counts = state[:12].sum(axis=(1, 2))
    white_scores = (piece_counts[0::2] * PIECE_VALUES).sum()
    black_scores = (piece_counts[1::2] * PIECE_VALUES).sum()
    return float(white_scores - black_scores)


def worker(remote, parent_remote):
    parent_remote.close()
    env = OpenSpielEnv()
    previous_state = None
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'close':
                break
            elif cmd == 'reset':
                env.reset()
                obs = env.state()
                previous_state = obs
                legal = env.get_legal_actions().numpy()
                remote.send((obs, legal))
            elif cmd == 'step':
                action = data
                env.step([action])

                obs = env.state()
                done = env.is_done()
                reward = _piece_difference_np(obs) - _piece_difference_np(previous_state)

                terminal_obs = None
                game_result = None
                if done:
                    terminal_obs = obs.copy()
                    game_result = env.game_result()
                    env.reset()
                    obs = env.state()

                previous_state = obs
                legal = env.get_legal_actions().numpy()
                remote.send((obs, reward, done, terminal_obs, game_result, legal))
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        remote.close()


class OpenSpielAsyncVectorEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.ctx = mp.get_context('fork')
        self.remotes, self.work_remotes = zip(*[self.ctx.Pipe() for _ in range(num_envs)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = self.ctx.Process(target=worker, args=(work_remote, remote))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self) -> EnvStep:
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, legal_actions = zip(*results)
        return EnvStep(
            obs=torch.tensor(np.stack(obs)).float(),
            legal_actions_mask=torch.tensor(np.stack(legal_actions)).float(),
            reward=torch.zeros(self.num_envs),
            done=torch.zeros(self.num_envs, dtype=torch.bool),
            info={},
        )

    def step(self, actions) -> EnvStep:
        if isinstance(actions, torch.Tensor):
            actions = actions.tolist()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]
        obs_list, rewards, dones, terminal_obs_list, game_result_list, legal_list = zip(*results)

        info = {}
        terminal_observations = {}
        game_results = {}
        for i, (t_obs, g_result) in enumerate(zip(terminal_obs_list, game_result_list)):
            if t_obs is not None:
                terminal_observations[i] = t_obs
                game_results[i] = g_result
        if terminal_observations:
            info["terminal_observations"] = terminal_observations
            info["game_results"] = game_results

        return EnvStep(
            obs=torch.tensor(np.stack(obs_list)).float(),
            legal_actions_mask=torch.tensor(np.stack(legal_list)).float(),
            reward=torch.tensor(rewards).float(),
            done=torch.tensor(dones).bool(),
            info=info,
        )

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/envs/test_open_spiel_async_vector_env.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/envs/open_spiel_async_vector_env.py tests/envs/test_open_spiel_async_vector_env.py
git commit -m "feat: create OpenSpielAsyncVectorEnv with EnvStep API"
```

### Task 4: Remove old files and VectorEnv ABC

**Files:**
- Delete: `src/environment/async_vector_env.py`
- Delete: `tests/test_async_env.py`
- Modify: `src/core/interfaces.py` — restore to empty or remove VectorEnv
- Modify: `src/envs/open_spiel_vector_env.py` — remove `VectorEnv` import (already done in Task 2)

**Step 1: Delete old files**

```bash
rm src/environment/async_vector_env.py
rm tests/test_async_env.py
```

**Step 2: Restore `src/core/interfaces.py`**

Replace contents with empty module (or delete if nothing else uses it):
```python
# Interfaces module — currently unused.
# VectorEnv ABC removed in favor of duck typing with EnvStep.
```

**Step 3: Run all env tests to verify nothing broke**

Run: `source .venv/bin/activate && python -m pytest tests/envs/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove VectorEnv ABC and old async_vector_env"
```
