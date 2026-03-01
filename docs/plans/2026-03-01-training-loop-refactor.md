# Training Loop Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `training.py` to use the `EnvStep` API from the new vectorized environments, then remove the legacy `EnvSpawner`.

**Architecture:** Player-agnostic A2C training loop in CleanRL style. Each step dispatches to `white_model` or `black_model` based on `EnvStep.current_player`. Returns computation handles cross-player `done` propagation via pure tensor operations. Terminal rewards configurable via YAML.

**Tech Stack:** PyTorch, OpenSpiel, existing loss modules (`ComposedLoss`, policy gradient, value MSE, entropy)

**Design doc:** `docs/plans/2026-03-01-training-loop-refactor-design.md`

---

### Task 1: Add `current_player` to `EnvStep`

**Files:**
- Modify: `src/core/types.py`
- Modify: `tests/core/test_interfaces_contracts.py`

**Step 1: Update the contract test to expect `current_player`**

```python
# tests/core/test_interfaces_contracts.py
def test_env_step_has_required_fields():
    names = {f.name for f in fields(EnvStep)}
    assert {"obs", "legal_actions_mask", "reward", "done", "current_player", "info"} <= names
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_interfaces_contracts.py::test_env_step_has_required_fields -v`
Expected: FAIL — `current_player` not in fields

**Step 3: Add `current_player` field to `EnvStep`**

```python
# src/core/types.py
@dataclass
class EnvStep:
    obs: Tensor
    legal_actions_mask: Tensor
    reward: Tensor
    done: Tensor
    current_player: Tensor  # [num_envs], 0=white, 1=black
    info: Dict[str, Any]
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_interfaces_contracts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/types.py tests/core/test_interfaces_contracts.py
git commit -m "feat: add current_player field to EnvStep"
```

---

### Task 2: Update `OpenSpielVectorEnv` to populate `current_player`

**Files:**
- Modify: `src/envs/open_spiel_vector_env.py`
- Modify: `tests/envs/test_open_spiel_vector_env.py`

**Step 1: Write test for `current_player` in reset and step**

```python
# tests/envs/test_open_spiel_vector_env.py — add new test
def test_vector_env_reset_returns_current_player():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    assert step.current_player.shape == (2,)
    assert (step.current_player == 0).all(), "After reset all envs should be white-to-move"


def test_vector_env_step_returns_current_player():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    legal = step.legal_actions_mask
    actions = []
    for i in range(2):
        legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
        actions.append(legal_indices[0].item())
    step = env.step(torch.tensor(actions))
    assert step.current_player.shape == (2,)
    assert (step.current_player == 1).all(), "After white moves, all envs should be black-to-move"
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/envs/test_open_spiel_vector_env.py::test_vector_env_reset_returns_current_player tests/envs/test_open_spiel_vector_env.py::test_vector_env_step_returns_current_player -v`
Expected: FAIL — `EnvStep.__init__() missing required argument: 'current_player'` (or TypeError)

**Step 3: Implement `current_player` in `OpenSpielVectorEnv`**

In `reset()` — add after `done = ...`:
```python
current_player = torch.zeros(self.num_envs, dtype=torch.long)
```
Pass to `EnvStep(... current_player=current_player ...)`.

In `step()` — add after the per-env loop, before building `EnvStep`:
```python
current_player = torch.tensor(
    [env.get_current_player() for env in self.envs], dtype=torch.long
)
```
Pass to `EnvStep(... current_player=current_player ...)`.

Note: `get_current_player()` returns -4 for terminal states. After auto-reset the env is fresh so this returns 0 (white). No special handling needed.

**Step 4: Run all vector env tests**

Run: `.venv/bin/python -m pytest tests/envs/test_open_spiel_vector_env.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/envs/open_spiel_vector_env.py tests/envs/test_open_spiel_vector_env.py
git commit -m "feat: populate current_player in OpenSpielVectorEnv"
```

---

### Task 3: Update `OpenSpielAsyncVectorEnv` to populate `current_player`

**Files:**
- Modify: `src/envs/open_spiel_async_vector_env.py`
- Modify: `tests/envs/test_open_spiel_async_vector_env.py`

**Step 1: Write test for `current_player` in reset and step**

```python
# tests/envs/test_open_spiel_async_vector_env.py — add new tests
def test_async_env_reset_returns_current_player():
    env = OpenSpielAsyncVectorEnv(num_envs=2)
    try:
        step = env.reset()
        assert step.current_player.shape == (2,)
        assert (step.current_player == 0).all()
    finally:
        env.close()


def test_async_env_step_returns_current_player():
    env = OpenSpielAsyncVectorEnv(num_envs=2)
    try:
        step = env.reset()
        legal = step.legal_actions_mask
        actions = []
        for i in range(2):
            legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
            actions.append(legal_indices[0].item())
        step = env.step(torch.tensor(actions))
        assert step.current_player.shape == (2,)
        assert (step.current_player == 1).all()
    finally:
        env.close()
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/envs/test_open_spiel_async_vector_env.py::test_async_env_reset_returns_current_player tests/envs/test_open_spiel_async_vector_env.py::test_async_env_step_returns_current_player -v`
Expected: FAIL

**Step 3: Implement in worker and master**

Worker `reset` handler — send `current_player` alongside obs and legal:
```python
elif cmd == "reset":
    env.reset()
    obs = env.state()
    previous_state = obs
    legal = env.get_legal_actions().numpy()
    cp = env.get_current_player()
    remote.send((obs, legal, cp))
```

Worker `step` handler — send `current_player` after auto-reset:
```python
elif cmd == "step":
    # ... existing step logic ...
    cp = env.get_current_player()
    remote.send((obs, reward, done, terminal_obs, game_result, legal, cp))
```

Master `reset()` — unpack and assemble:
```python
def reset(self) -> EnvStep:
    # ...
    results = [remote.recv() for remote in self.remotes]
    obs, legal_actions, current_players = zip(*results)
    return EnvStep(
        # ... existing fields ...
        current_player=torch.tensor(current_players, dtype=torch.long),
        # ...
    )
```

Master `step()` — unpack and assemble:
```python
def step(self, actions) -> EnvStep:
    # ...
    results = [remote.recv() for remote in self.remotes]
    obs_list, rewards, dones, terminal_obs_list, game_result_list, legal_list, cp_list = zip(*results)
    # ...
    return EnvStep(
        # ... existing fields ...
        current_player=torch.tensor(cp_list, dtype=torch.long),
        # ...
    )
```

**Step 4: Run all async env tests**

Run: `.venv/bin/python -m pytest tests/envs/test_open_spiel_async_vector_env.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/envs/open_spiel_async_vector_env.py tests/envs/test_open_spiel_async_vector_env.py
git commit -m "feat: populate current_player in OpenSpielAsyncVectorEnv"
```

---

### Task 4: `StepRecord` and `compute_returns_for_model`

This is the core algorithm. Thoroughly tested because cross-player done propagation is subtle.

**Files:**
- Modify: `src/core/types.py` (add `StepRecord`)
- Create: `src/model/returns.py`
- Create: `tests/model/test_returns.py`

**Step 1: Add `StepRecord` to types**

```python
# src/core/types.py — add after EnvStep
@dataclass
class StepRecord:
    player: Tensor           # [num_envs], 0=white, 1=black
    value: Tensor            # [num_envs, 1]
    log_prob: Tensor         # [num_envs, 1]
    entropy: Tensor          # [num_envs, 1]
    reward_white: Tensor     # [num_envs], piece-diff from white's perspective
    done: Tensor             # [num_envs], bool
    terminal_r_white: Tensor # [num_envs], precomputed terminal reward for white
    terminal_r_black: Tensor # [num_envs], precomputed terminal reward for black
```

**Step 2: Write test — single player, no done**

```python
# tests/model/test_returns.py
import torch
from torch.testing import assert_close
from src.core.types import StepRecord
from src.model.returns import compute_returns_for_model


def _make_step(player, reward_white, done, terminal_r_w=0.0, terminal_r_b=0.0, n=1):
    return StepRecord(
        player=torch.tensor([player] * n, dtype=torch.long),
        value=torch.zeros(n, 1),
        log_prob=torch.zeros(n, 1),
        entropy=torch.zeros(n, 1),
        reward_white=torch.tensor([reward_white] * n),
        done=torch.tensor([done] * n),
        terminal_r_white=torch.tensor([terminal_r_w] * n),
        terminal_r_black=torch.tensor([terminal_r_b] * n),
    )


def test_returns_single_player_no_done():
    """All white steps, no done. Standard discounted returns."""
    steps = [
        _make_step(player=0, reward_white=1.0, done=False),
        _make_step(player=0, reward_white=2.0, done=False),
        _make_step(player=0, reward_white=3.0, done=False),
    ]
    bootstrap = torch.tensor([10.0])
    returns = compute_returns_for_model(steps, model_id=0, bootstrap_value=bootstrap, gamma=0.99)
    # R_2 = 3 + 0.99 * 10 = 12.9
    # R_1 = 2 + 0.99 * 12.9 = 14.771
    # R_0 = 1 + 0.99 * 14.771 = 15.62329
    assert_close(returns[2], torch.tensor([12.9]))
    assert_close(returns[1], torch.tensor([14.771]))
    assert_close(returns[0], torch.tensor([15.62329]))
```

**Step 3: Write test — alternating players, no done**

```python
# tests/model/test_returns.py — add
def test_returns_alternating_no_done():
    """White and black alternate. White's return skips black's steps."""
    steps = [
        _make_step(player=0, reward_white=1.0, done=False),   # white
        _make_step(player=1, reward_white=-2.0, done=False),  # black
        _make_step(player=0, reward_white=3.0, done=False),   # white
    ]
    bootstrap = torch.tensor([0.0])
    returns = compute_returns_for_model(steps, model_id=0, bootstrap_value=bootstrap, gamma=1.0)
    # gamma=1.0 for simplicity
    # Step 2 (white): R = 3 + 1.0 * 0 = 3
    # Step 1 (black): is_mine=False, R stays 3
    # Step 0 (white): R = 1 + 1.0 * 3 = 4
    assert_close(returns[2], torch.tensor([3.0]))
    assert_close(returns[0], torch.tensor([4.0]))
```

**Step 4: Write test — done on own step cuts chain**

```python
# tests/model/test_returns.py — add
def test_returns_done_on_own_step():
    """Done on white's step cuts the chain."""
    steps = [
        _make_step(player=0, reward_white=1.0, done=False),
        _make_step(player=0, reward_white=5.0, done=True, terminal_r_w=2.0),
        _make_step(player=0, reward_white=0.0, done=False),  # new game
    ]
    bootstrap = torch.tensor([10.0])
    returns = compute_returns_for_model(steps, model_id=0, bootstrap_value=bootstrap, gamma=1.0)
    # Step 2 (new game): R = 0 + 1.0 * 10 = 10
    # Step 1 (done=T):
    #   A: R = 10 * (1-1) + 2.0 = 2.0  (chain cut, terminal reward injected)
    #   B: is_mine=True → R = 5 + 1.0 * 2.0 = 7.0
    # Step 0: R = 1 + 1.0 * 7.0 = 8.0
    assert_close(returns[2], torch.tensor([10.0]))
    assert_close(returns[1], torch.tensor([7.0]))
    assert_close(returns[0], torch.tensor([8.0]))
```

**Step 5: Write test — done on OPPONENT's step cuts chain (critical test)**

```python
# tests/model/test_returns.py — add
def test_returns_cross_player_done():
    """Black checkmates at step 1. White's return at step 0 must NOT include step 2 (new game)."""
    steps = [
        _make_step(player=0, reward_white=0.0, done=False),
        _make_step(player=1, reward_white=0.0, done=True, terminal_r_w=-2.0, terminal_r_b=2.0),
        _make_step(player=0, reward_white=0.0, done=False),  # new game after reset
    ]
    bootstrap = torch.tensor([5.0])
    returns = compute_returns_for_model(steps, model_id=0, bootstrap_value=bootstrap, gamma=1.0)
    # Step 2 (white, new game): R = 0 + 1.0 * 5.0 = 5.0
    # Step 1 (black, done=T):
    #   A: R = 5.0 * (1-1) + (-2.0) = -2.0  (chain cut!)
    #   B: is_mine=False → R stays -2.0
    # Step 0 (white): R = 0 + 1.0 * (-2.0) = -2.0
    assert_close(returns[2], torch.tensor([5.0]))
    assert_close(returns[0], torch.tensor([-2.0]))
```

**Step 6: Write test — black model's perspective**

```python
# tests/model/test_returns.py — add
def test_returns_black_model_perspective():
    """Rewards are flipped for black model."""
    steps = [
        _make_step(player=0, reward_white=1.0, done=False),
        _make_step(player=1, reward_white=-3.0, done=False),  # from white persp; black gets +3
    ]
    bootstrap = torch.tensor([0.0])
    returns = compute_returns_for_model(steps, model_id=1, bootstrap_value=bootstrap, gamma=1.0)
    # Step 1 (black): reward_for_black = -(-3.0) = 3.0 → R = 3.0 + 0 = 3.0
    # Step 0 (white): is_mine=False → R stays 3.0
    assert_close(returns[1], torch.tensor([3.0]))
```

**Step 7: Run all tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_returns.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.returns'`

**Step 8: Implement `compute_returns_for_model`**

```python
# src/model/returns.py
import torch
from src.core.types import StepRecord


def compute_returns_for_model(
    steps: list[StepRecord],
    model_id: int,
    bootstrap_value: torch.Tensor,
    gamma: float,
) -> list[torch.Tensor]:
    """Compute discounted returns for a specific model (0=white, 1=black).

    Handles cross-player done propagation: if the game ends on the
    opponent's step, the return chain is still cut correctly.
    """
    R = bootstrap_value.clone()
    returns = [None] * len(steps)

    for i in reversed(range(len(steps))):
        step = steps[i]
        is_mine = (step.player == model_id)
        done_f = step.done.float()

        terminal_r = step.terminal_r_white if model_id == 0 else step.terminal_r_black
        R = R * (1.0 - done_f) + terminal_r

        reward = step.reward_white if model_id == 0 else -step.reward_white
        new_R = reward + gamma * R
        R = torch.where(is_mine, new_R, R)

        returns[i] = R.clone()

    return returns
```

**Step 9: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_returns.py -v`
Expected: ALL PASS

**Step 10: Commit**

```bash
git add src/core/types.py src/model/returns.py tests/model/test_returns.py tests/model/__init__.py
git commit -m "feat: add StepRecord and compute_returns_for_model with cross-player done"
```

---

### Task 5: Rewrite `training.py`

**Files:**
- Rewrite: `src/model/training.py`
- Modify: `tests/integration/test_training_smoke.py`

**Step 1: Update smoke test for new config shape**

The smoke test calls `run_training_from_config` which will change. Update it to pass new config keys:

```python
# tests/integration/test_training_smoke.py
import torch
from src.entrypoints.train import run_training_from_config


def test_training_smoke_runs_short_job_without_nan():
    config = {
        "num_envs": 2,
        "max_updates": 12,
        "steps_per_update": 3,
        "seed": 42,
        "env_type": "sync",
        "terminal_rewards": {"win": 2.0, "loss": -2.0, "draw": -0.5},
    }
    result = run_training_from_config(config)

    assert result is not None
    assert "logs" in result
    assert "losses" in result
    assert len(result["losses"]) > 0, "No losses recorded"

    for loss in result["losses"]:
        assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
```

**Step 2: Run smoke test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_training_smoke.py -v`
Expected: FAIL (old training.py + old entrypoint)

**Step 3: Rewrite `training.py`**

Full replacement of `src/model/training.py`. Key structure:

```python
# src/model/training.py
import torch
from tqdm import tqdm
from torch.distributions import Categorical
from src.core.types import EnvStep, StepRecord
from src.model.returns import compute_returns_for_model
from src.training.metrics import MetricsAggregator
from src.losses.policy_gradient import compute_policy_gradient_loss
from src.losses.value_mse import compute_value_loss
from src.losses.entropy_regularization import compute_entropy_bonus
from src.losses.composed_loss import ComposedLoss

RESULT_TO_REWARD = {
    "white_win": {"white": "win", "black": "loss"},
    "black_win": {"white": "loss", "black": "win"},
    "draw":      {"white": "draw", "black": "draw"},
}
MODEL_NAMES = {0: "white", 1: "black"}


def _precompute_terminal_rewards(info, num_envs, terminal_rewards):
    """Convert info['game_results'] into per-model reward tensors."""
    tr_white = torch.zeros(num_envs)
    tr_black = torch.zeros(num_envs)
    game_results = info.get("game_results", {})
    for env_idx, result in game_results.items():
        mapping = RESULT_TO_REWARD[result]
        tr_white[env_idx] = terminal_rewards[mapping["white"]]
        tr_black[env_idx] = terminal_rewards[mapping["black"]]
    return tr_white, tr_black


def _collect_rollout(envs, white_model, black_model, env_step, num_steps, num_envs,
                     terminal_rewards, metrics):
    """Collect num_steps of player-agnostic experience."""
    models = {0: white_model, 1: black_model}
    steps_data = []

    for _ in range(num_steps):
        players = env_step.current_player
        white_mask = (players == 0)
        black_mask = (players == 1)

        probs = torch.zeros(num_envs, 4674)
        values = torch.zeros(num_envs, 1)

        for model_id, mask in [(0, white_mask), (1, black_mask)]:
            if not mask.any():
                continue
            model = models[model_id]
            p, v = model(env_step.obs[mask], env_step.legal_actions_mask[mask])
            probs[mask] = p
            values[mask] = v

        dist = Categorical(probs=probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).unsqueeze(1)
        entropies = dist.entropy().unsqueeze(1)

        # Diagnostics
        legal_moves_count = env_step.legal_actions_mask.sum(dim=1).float()
        log_legal_moves = torch.log(torch.clamp(legal_moves_count, min=2.0))
        normalized_entropy = dist.entropy() / log_legal_moves

        sampled_is_legal = env_step.legal_actions_mask.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1) > 0
        empty_masks = (env_step.legal_actions_mask.sum(dim=1) == 0).sum().item()
        illegal_samples = (~sampled_is_legal).sum().item()
        metrics.add_step(empty_masks=empty_masks, illegal_samples=illegal_samples)

        env_step = envs.step(actions)

        tr_white, tr_black = _precompute_terminal_rewards(
            env_step.info, num_envs, terminal_rewards
        )

        if "game_results" in env_step.info:
            for env_idx, result in env_step.info["game_results"].items():
                if result == "white_win":
                    metrics.add_terminal_result(white_win=True)
                elif result == "black_win":
                    metrics.add_terminal_result(black_win=True)
                else:
                    metrics.add_terminal_result(draw=True)

        steps_data.append(StepRecord(
            player=players.clone(),
            value=values.detach(),
            log_prob=log_probs,
            entropy=entropies,
            reward_white=env_step.reward.clone(),
            done=env_step.done.clone(),
            terminal_r_white=tr_white,
            terminal_r_black=tr_black,
        ))

    return steps_data, env_step, normalized_entropy


def _compute_bootstrap(env_step, white_model, black_model, model_id):
    """Bootstrap value from the perspective of model_id."""
    num_envs = env_step.obs.shape[0]
    bootstrap = torch.zeros(num_envs)
    with torch.no_grad():
        same_player = (env_step.current_player == model_id)
        opponent_id = 1 - model_id
        opponent_player = (env_step.current_player == opponent_id)
        models = {0: white_model, 1: black_model}

        if same_player.any():
            _, v = models[model_id](
                env_step.obs[same_player],
                env_step.legal_actions_mask[same_player],
            )
            bootstrap[same_player] = v.squeeze(-1)

        if opponent_player.any():
            _, v = models[opponent_id](
                env_step.obs[opponent_player],
                env_step.legal_actions_mask[opponent_player],
            )
            bootstrap[opponent_player] = -v.squeeze(-1)

    return bootstrap


def _backpropagate_for_model(model, optimizer, steps_data, model_returns, model_id):
    """Compute A2C losses and update model for the given player."""
    filtered_log_probs = []
    filtered_advantages = []
    filtered_values = []
    filtered_returns = []
    filtered_entropies = []

    for i, step in enumerate(steps_data):
        is_mine = (step.player == model_id)
        if not is_mine.any():
            continue
        advantage = (model_returns[i][is_mine] - step.value.squeeze(-1)[is_mine]).detach()
        filtered_advantages.append(advantage)
        filtered_log_probs.append(step.log_prob.squeeze(-1)[is_mine])
        filtered_values.append(step.value.squeeze(-1)[is_mine])
        filtered_returns.append(model_returns[i][is_mine].detach())
        filtered_entropies.append(step.entropy.squeeze(-1)[is_mine])

    if not filtered_log_probs:
        return torch.tensor(0.0)

    policy_loss = compute_policy_gradient_loss(filtered_log_probs, filtered_advantages)
    value_loss = compute_value_loss(filtered_values, filtered_returns)
    entropy_bonus = compute_entropy_bonus(filtered_entropies)

    loss_composer = ComposedLoss()
    loss_dict = loss_composer.compute(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        entropy_coef=0.01,
    )

    total_loss = loss_dict["total_loss"]
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
    optimizer.step()
    return total_loss


def run_chess_training(
    envs,
    white_model,
    black_model,
    optimizer_white,
    optimizer_black,
    metrics: MetricsAggregator = None,
    episodes=2000,
    steps=5,
    lr_decay_interval=100,
    terminal_rewards=None,
):
    if metrics is None:
        metrics = MetricsAggregator()
    if terminal_rewards is None:
        terminal_rewards = {"win": 2.0, "loss": -2.0, "draw": -0.5}

    num_envs = envs.num_envs
    logs = []
    losses = []
    env_step = envs.reset()
    pbar = tqdm(range(1, episodes))

    for episode in pbar:
        steps_data, env_step, last_norm_entropy = _collect_rollout(
            envs, white_model, black_model, env_step, steps, num_envs,
            terminal_rewards, metrics,
        )

        bootstrap_white = _compute_bootstrap(env_step, white_model, black_model, model_id=0)
        bootstrap_black = _compute_bootstrap(env_step, white_model, black_model, model_id=1)

        returns_white = compute_returns_for_model(steps_data, model_id=0,
                                                   bootstrap_value=bootstrap_white, gamma=0.99)
        returns_black = compute_returns_for_model(steps_data, model_id=1,
                                                   bootstrap_value=bootstrap_black, gamma=0.99)

        loss_w = _backpropagate_for_model(white_model, optimizer_white,
                                          steps_data, returns_white, model_id=0)
        loss_b = _backpropagate_for_model(black_model, optimizer_black,
                                          steps_data, returns_black, model_id=1)

        total_loss = loss_w.item() + loss_b.item()
        losses.append(total_loss)

        pbar.set_postfix(loss=total_loss)

        if episode % lr_decay_interval == 0:
            for opt in [optimizer_white, optimizer_black]:
                opt.param_groups[0]["lr"] = max(3e-4, opt.param_groups[0]["lr"] * 0.5)

        if episode % 10 == 0:
            summary = metrics.episode_summary()
            logs.append({"episode": episode, "loss": total_loss, **summary})

    if hasattr(envs, 'close'):
        envs.close()

    return logs, losses, white_model, black_model
```

**Step 4: Update `src/entrypoints/train.py`**

```python
# src/entrypoints/train.py
import torch
import random
import numpy as np
from typing import Dict, Any
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _create_envs(env_type: str, num_envs: int):
    if env_type == "async":
        return OpenSpielAsyncVectorEnv(num_envs)
    elif env_type == "sync":
        return OpenSpielVectorEnv(num_envs)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def run_training_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    num_envs = config.get("num_envs", 12)
    max_updates = config.get("max_updates", 30000)
    steps = config.get("steps_per_update", 15)
    lr = config.get("learning_rate", 1e-4)
    seed = config.get("seed", 42)
    lr_decay_interval = config.get("lr_decay_interval", 100)
    env_type = config.get("env_type", "sync")
    terminal_rewards = config.get("terminal_rewards", {"win": 2.0, "loss": -2.0, "draw": -0.5})

    set_seed(seed)

    envs = _create_envs(env_type, num_envs)
    white_model = ChessPolicyProbs()
    black_model = ChessPolicyProbs()
    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=lr)
    optimizer_black = torch.optim.Adam(black_model.parameters(), lr=lr)

    logs, losses, white_model, black_model = run_chess_training(
        envs, white_model, black_model,
        optimizer_white, optimizer_black,
        steps=steps, episodes=max_updates,
        lr_decay_interval=lr_decay_interval,
        terminal_rewards=terminal_rewards,
    )

    return {
        "logs": logs,
        "losses": losses,
        "white_model": white_model,
        "black_model": black_model,
    }


def main():
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Train chess RL agent")
    parser.add_argument("--config", type=str, default="src/configs/train_default.yaml")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--env-type", type=str, choices=["sync", "async"])

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.max_updates:
        config["max_updates"] = args.max_updates
    if args.seed:
        config["seed"] = args.seed
    if args.env_type:
        config["env_type"] = args.env_type

    result = run_training_from_config(config)
    print(f"Training complete. Processed {len(result['logs'])} log entries.")
    return result


if __name__ == "__main__":
    main()
```

**Step 5: Update `src/main.py`**

```python
# src/main.py
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json
from src.entrypoints.train import run_training_from_config

if __name__ == "__main__":
    config = {
        "num_envs": 12,
        "max_updates": 500,
        "steps_per_update": 15,
        "learning_rate": 1e-4,
        "seed": 42,
        "lr_decay_interval": 10001,
        "env_type": "sync",
        "terminal_rewards": {"win": 2.0, "loss": -2.0, "draw": -0.5},
    }

    result = run_training_from_config(config)

    logs = result["logs"]
    losses = result["losses"]
    white_model = result["white_model"]
    black_model = result["black_model"]

    plt.plot(losses)
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.savefig("losses.png")
    plt.close()

    with open("logs.json", "w") as f:
        json.dump(logs, f, indent=4)

    episodes = config["max_updates"]
    torch.save(
        white_model.state_dict(),
        f"white_model_{datetime.now().strftime('%Y%m%d%H%M%S')}_episodes_{episodes}.pth",
    )
    torch.save(
        black_model.state_dict(),
        f"black_model_{datetime.now().strftime('%Y%m%d%H%M%S')}_episodes_{episodes}.pth",
    )
    print("Training complete.")
```

**Step 6: Run smoke test**

Run: `.venv/bin/python -m pytest tests/integration/test_training_smoke.py -v`
Expected: PASS

**Step 7: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/core/test_interfaces_contracts.py`
Note: ignore contract test temporarily if `RolloutBatch` test breaks.
Expected: ALL PASS (or known pre-existing failures only)

**Step 8: Commit**

```bash
git add src/model/training.py src/entrypoints/train.py src/main.py tests/integration/test_training_smoke.py
git commit -m "feat: rewrite training loop for EnvStep API with player-agnostic A2C"
```

---

### Task 6: Cleanup

**Files:**
- Move: `VectorBuffer` from `src/environment/environment.py` → `src/training/vector_buffer.py`
- Delete: `src/environment/environment.py`
- Delete: `src/core/interfaces.py`
- Modify: `src/utils/utils.py` (remove unused `piece_difference_from_tensor`)

**Step 1: Move `VectorBuffer` to standalone module**

Copy the `VectorBuffer` class (lines 8-50 of `src/environment/environment.py`) to `src/training/vector_buffer.py` with its imports (`torch`, `random`).

**Step 2: Delete legacy files**

```bash
rm src/environment/environment.py
rm src/core/interfaces.py
```

**Step 3: Remove `piece_difference_from_tensor` from utils**

Remove the function from `src/utils/utils.py`. Keep `board_from_shaped_observation` (used by notebooks).

**Step 4: Verify no broken imports**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

If any test imports `EnvSpawner` or `VectorBuffer` from old paths, fix them.

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove legacy EnvSpawner, move VectorBuffer to training module"
```

---

### Task 7: Final verification

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Run a short training to verify end-to-end**

Run: `.venv/bin/python -c "from src.entrypoints.train import run_training_from_config; r = run_training_from_config({'num_envs': 4, 'max_updates': 20, 'steps_per_update': 5, 'seed': 42, 'env_type': 'sync', 'terminal_rewards': {'win': 2.0, 'loss': -2.0, 'draw': -0.5}}); print(f'OK: {len(r[\"losses\"])} updates')"` 
Expected: `OK: 19 updates` (or similar, no crashes)

**Step 3: Run with async env**

Run: `.venv/bin/python -c "from src.entrypoints.train import run_training_from_config; r = run_training_from_config({'num_envs': 4, 'max_updates': 20, 'steps_per_update': 5, 'seed': 42, 'env_type': 'async', 'terminal_rewards': {'win': 2.0, 'loss': -2.0, 'draw': -0.5}}); print(f'OK: {len(r[\"losses\"])} updates')"` 
Expected: `OK: 19 updates` (no crashes, no hanging)
