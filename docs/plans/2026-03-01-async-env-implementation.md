# Async Vector Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a vectorized environment wrapper (`AsyncVectorEnv`) that runs multiple `OpenSpielEnv` instances in parallel CPU processes to speed up data collection.

**Architecture:** We will define an abstract `VectorEnv` interface and a concrete `AsyncVectorEnv` implementation using `torch.multiprocessing` and `Pipe`. The environment will handle auto-resetting and return terminal observations in the `info` dictionary.

**Tech Stack:** Python, PyTorch (multiprocessing), OpenSpiel.

---

### Task 1: Define VectorEnv Interface

**Files:**
- Modify: `src/core/interfaces.py`

**Step 1: Write the failing test**

Create a temporary test file `tests/test_interfaces.py` to verify the interface exists (or just rely on the fact that we can't import it yet). Since this is an interface definition, we can skip a formal "failing test" for behavior, but we should ensure the file structure is correct.

**Step 2: Write minimal implementation**

```python
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Dict
import torch

class VectorEnv(ABC):
    """Abstract base class for vectorized environments."""
    
    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset all environments and return initial observations.
        Returns:
            observations (torch.Tensor): Batch of initial observations. Shape: (num_envs, *obs_shape)
            legal_actions (torch.Tensor): Batch of legal action masks. Shape: (num_envs, num_actions)
        """
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]], torch.Tensor]:
        """
        Step all environments with the given actions.
        Args:
            actions (torch.Tensor): Batch of actions. Shape: (num_envs, *action_shape)
        Returns:
            Tuple containing:
                - next_states (torch.Tensor): Batch of next observations.
                - rewards (torch.Tensor): Batch of rewards.
                - dones (torch.Tensor): Batch of done flags.
                - infos (List[Dict[str, Any]]): Auxiliary information (including terminal observations).
                - legal_actions (torch.Tensor): Batch of legal action masks.
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
```

**Step 3: Commit**

```bash
git add src/core/interfaces.py
git commit -m "feat: define VectorEnv interface"
```

### Task 2: Implement AsyncVectorEnv Skeleton & Worker

**Files:**
- Create: `src/environment/async_vector_env.py`
- Create: `tests/test_async_env.py`

**Step 1: Write the failing test**

```python
import pytest
import torch
from src.environment.async_vector_env import AsyncVectorEnv

def test_async_env_initialization():
    num_envs = 2
    env = AsyncVectorEnv(num_envs=num_envs)
    assert env.num_envs == num_envs
    env.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_async_env.py`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
import torch.multiprocessing as mp
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from src.core.interfaces import VectorEnv
from src.envs.open_spiel_env import OpenSpielEnv
from src.utils.utils import piece_difference_from_tensor

def worker(remote, parent_remote):
    parent_remote.close()
    env = OpenSpielEnv()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'close':
                break
            # ... other commands later
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        remote.close()

class AsyncVectorEnv(VectorEnv):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.ctx = mp.get_context('spawn') # Safer for CUDA
        self.remotes, self.work_remotes = zip(*[self.ctx.Pipe() for _ in range(num_envs)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = self.ctx.Process(target=worker, args=(work_remote, remote))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self): pass
    def step(self, actions): pass
    
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_async_env.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/environment/async_vector_env.py tests/test_async_env.py
git commit -m "feat: implement AsyncVectorEnv skeleton"
```

### Task 3: Implement Reset Logic

**Files:**
- Modify: `src/environment/async_vector_env.py`
- Modify: `tests/test_async_env.py`

**Step 1: Write the failing test**

```python
def test_async_env_reset():
    env = AsyncVectorEnv(num_envs=2)
    obs, legal_actions = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (2, 20, 8, 8)
    assert isinstance(legal_actions, torch.Tensor)
    assert legal_actions.shape == (2, 4674)
    env.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_async_env.py`
Expected: FAIL (NotImplementedError or NoneType)

**Step 3: Write minimal implementation**

Update `worker` to handle 'reset':
```python
            elif cmd == 'reset':
                env.reset()
                remote.send((env.state(), env.get_legal_actions()))
```

Update `AsyncVectorEnv.reset`:
```python
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, legal_actions = zip(*results)
        return torch.tensor(np.stack(obs)).float(), torch.stack(legal_actions)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_async_env.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/environment/async_vector_env.py tests/test_async_env.py
git commit -m "feat: implement AsyncVectorEnv reset"
```

### Task 4: Implement Step Logic with Auto-Reset

**Files:**
- Modify: `src/environment/async_vector_env.py`
- Modify: `tests/test_async_env.py`

**Step 1: Write the failing test**

```python
def test_async_env_step():
    env = AsyncVectorEnv(num_envs=2)
    obs, legal = env.reset()
    
    # Create dummy actions (just pick first legal action)
    actions = []
    for i in range(2):
        legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
        actions.append(legal_indices[0].item())
    
    actions_tensor = torch.tensor(actions)
    next_obs, rewards, dones, infos, next_legal = env.step(actions_tensor)
    
    assert next_obs.shape == (2, 20, 8, 8)
    assert rewards.shape == (2, 1)
    assert dones.shape == (2, 1)
    assert len(infos) == 2
    assert next_legal.shape == (2, 4674)
    env.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_async_env.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `worker` to handle 'step' with auto-reset:
```python
            elif cmd == 'step':
                action = data
                env.step(action)
                
                obs = env.state()
                done = env.is_done()
                
                # Calculate reward (simplified for now, using piece difference)
                # Note: We need previous state for accurate reward calc, let's add that tracking
                # For now, just return 0.0 or implement the diff logic
                reward = 0.0 
                
                info = {}
                if done:
                    info["terminal_observation"] = obs
                    env.reset()
                    obs = env.state()
                
                remote.send((obs, reward, done, info, env.get_legal_actions()))
```

Update `AsyncVectorEnv.step`:
```python
    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.tolist()
            
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos, legal_actions = zip(*results)
        
        return (
            torch.tensor(np.stack(obs)).float(),
            torch.tensor(rewards).float().unsqueeze(1),
            torch.tensor(dones).bool().unsqueeze(1),
            list(infos),
            torch.stack(legal_actions)
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_async_env.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/environment/async_vector_env.py tests/test_async_env.py
git commit -m "feat: implement AsyncVectorEnv step with auto-reset"
```

### Task 5: Implement Proper Reward Calculation

**Files:**
- Modify: `src/environment/async_vector_env.py`

**Step 1: Write the failing test**
(Refine existing test or add new one to check non-zero rewards)

**Step 2: Write implementation**
Integrate `piece_difference_from_tensor` logic inside the worker. The worker needs to track `previous_state`.

```python
# Inside worker
previous_state = env.state()
# ... inside loop ...
# Calculate reward using piece_difference_from_tensor logic on numpy arrays or tensors
```

**Step 3: Verify & Commit**

```bash
git add src/environment/async_vector_env.py
git commit -m "feat: implement proper reward calculation in worker"
```
