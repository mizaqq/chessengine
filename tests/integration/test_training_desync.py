"""Test that the training loop handles desynchronized and edge-case environments.

Covers:
  - Single env done (mixed current_player batch)
  - All envs done simultaneously (100% terminal reward, full reset)
  - Both models receive gradients in all scenarios
"""
import torch
from src.core.types import EnvStep
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training


class DesyncVectorEnv:
    """Mock env that forces desynchronization after a few steps.

    Env 0 terminates (done=True) on step `done_at`, causing it to
    reset to white-to-move while other envs may be on black's turn.
    """

    def __init__(self, num_envs=4, done_at=3):
        self.num_envs = num_envs
        self._done_at = done_at
        self._step_count = 0
        self._current_player = torch.ones(num_envs, dtype=torch.long)

    def reset(self) -> EnvStep:
        self._step_count = 0
        self._current_player = torch.ones(self.num_envs, dtype=torch.long)
        return self._make_env_step(done=False, game_result=None, reset_idx=None)

    def step(self, actions) -> EnvStep:
        self._step_count += 1
        self._current_player = 1 - self._current_player

        force_done = (self._step_count == self._done_at)
        if force_done:
            self._current_player[0] = 1
            return self._make_env_step(
                done=True, game_result="black_win", reset_idx=0,
            )

        return self._make_env_step(done=False, game_result=None, reset_idx=None)

    def _make_env_step(self, done, game_result, reset_idx):
        obs = torch.randn(self.num_envs, 20, 8, 8)
        legal = torch.zeros(self.num_envs, 4674)
        legal[:, :20] = 1.0

        done_tensor = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.zeros(self.num_envs)

        info = {}
        if done and reset_idx is not None:
            done_tensor[reset_idx] = True
            info["game_results"] = {reset_idx: game_result}
            info["terminal_observations"] = {reset_idx: obs[reset_idx].numpy()}

        return EnvStep(
            obs=obs,
            legal_actions_mask=legal,
            reward=reward,
            done=done_tensor,
            current_player=self._current_player.clone(),
            info=info,
        )


def test_training_survives_desynchronized_envs():
    """Training loop must handle mixed current_player batches without crashing."""
    envs = DesyncVectorEnv(num_envs=4, done_at=3)
    white_model = ChessPolicyProbs(num_filters=16)
    black_model = ChessPolicyProbs(num_filters=16)
    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=1e-4)
    optimizer_black = torch.optim.Adam(black_model.parameters(), lr=1e-4)

    logs, losses, _, _ = run_chess_training(
        envs, white_model, black_model,
        optimizer_white, optimizer_black,
        episodes=10,
        steps=5,
        terminal_rewards={"win": 2.0, "loss": -2.0, "draw": -0.5},
    )

    assert len(losses) == 10, f"Expected 10 updates, got {len(losses)}"
    for i, loss in enumerate(losses):
        assert not torch.isnan(torch.tensor(loss)), f"NaN loss at update {i}"


def test_training_produces_gradients_for_both_models_with_desync():
    """Both value heads must receive gradients when envs are desynchronized."""
    envs = DesyncVectorEnv(num_envs=4, done_at=2)
    white_model = ChessPolicyProbs(num_filters=16)
    black_model = ChessPolicyProbs(num_filters=16)

    w_params_before = {n: p.clone() for n, p in white_model.named_parameters()}
    b_params_before = {n: p.clone() for n, p in black_model.named_parameters()}

    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=1e-3)
    optimizer_black = torch.optim.Adam(black_model.parameters(), lr=1e-3)

    run_chess_training(
        envs, white_model, black_model,
        optimizer_white, optimizer_black,
        episodes=5,
        steps=5,
        terminal_rewards={"win": 2.0, "loss": -2.0, "draw": -0.5},
    )

    white_changed = any(
        not torch.equal(w_params_before[n], p)
        for n, p in white_model.named_parameters()
    )
    black_changed = any(
        not torch.equal(b_params_before[n], p)
        for n, p in black_model.named_parameters()
    )
    assert white_changed, "White model parameters did not change — no gradients?"
    assert black_changed, "Black model parameters did not change — no gradients?"


class AllDoneVectorEnv:
    """Mock env where ALL envs terminate simultaneously every `done_every` steps.

    This is the degenerate case: 100% of the batch receives terminal rewards
    at once, then all envs restart synchronized (all white-to-move).
    """

    def __init__(self, num_envs=4, done_every=4):
        self.num_envs = num_envs
        self._done_every = done_every
        self._step_count = 0
        self._current_player = torch.ones(num_envs, dtype=torch.long)

    def reset(self) -> EnvStep:
        self._step_count = 0
        self._current_player = torch.ones(self.num_envs, dtype=torch.long)
        return self._make_env_step(all_done=False)

    def step(self, actions) -> EnvStep:
        self._step_count += 1
        self._current_player = 1 - self._current_player

        if self._step_count % self._done_every == 0:
            self._current_player[:] = 1
            return self._make_env_step(all_done=True)

        return self._make_env_step(all_done=False)

    def _make_env_step(self, all_done):
        obs = torch.randn(self.num_envs, 20, 8, 8)
        legal = torch.zeros(self.num_envs, 4674)
        legal[:, :20] = 1.0
        reward = torch.zeros(self.num_envs)

        done_tensor = torch.zeros(self.num_envs, dtype=torch.bool)
        info = {}
        if all_done:
            done_tensor[:] = True
            results = {}
            for i in range(self.num_envs):
                results[i] = "white_win" if i % 2 == 0 else "black_win"
            info["game_results"] = results
            info["terminal_observations"] = {
                i: obs[i].numpy() for i in range(self.num_envs)
            }

        return EnvStep(
            obs=obs,
            legal_actions_mask=legal,
            reward=reward,
            done=done_tensor,
            current_player=self._current_player.clone(),
            info=info,
        )


def test_training_survives_all_envs_done_simultaneously():
    """All envs finishing at once must not crash or produce NaN."""
    envs = AllDoneVectorEnv(num_envs=4, done_every=3)
    white_model = ChessPolicyProbs(num_filters=16)
    black_model = ChessPolicyProbs(num_filters=16)
    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=1e-4)
    optimizer_black = torch.optim.Adam(black_model.parameters(), lr=1e-4)

    logs, losses, _, _ = run_chess_training(
        envs, white_model, black_model,
        optimizer_white, optimizer_black,
        episodes=10,
        steps=6,
        terminal_rewards={"win": 2.0, "loss": -2.0, "draw": -0.5},
    )

    assert len(losses) == 10, f"Expected 10 updates, got {len(losses)}"
    for i, loss in enumerate(losses):
        assert not torch.isnan(torch.tensor(loss)), f"NaN loss at update {i}"


def test_all_done_both_models_receive_gradients():
    """Both models must learn even when all envs reset together."""
    envs = AllDoneVectorEnv(num_envs=4, done_every=2)
    white_model = ChessPolicyProbs(num_filters=16)
    black_model = ChessPolicyProbs(num_filters=16)

    w_params_before = {n: p.clone() for n, p in white_model.named_parameters()}
    b_params_before = {n: p.clone() for n, p in black_model.named_parameters()}

    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=1e-3)
    optimizer_black = torch.optim.Adam(black_model.parameters(), lr=1e-3)

    run_chess_training(
        envs, white_model, black_model,
        optimizer_white, optimizer_black,
        episodes=5,
        steps=6,
        terminal_rewards={"win": 2.0, "loss": -2.0, "draw": -0.5},
    )

    white_changed = any(
        not torch.equal(w_params_before[n], p)
        for n, p in white_model.named_parameters()
    )
    black_changed = any(
        not torch.equal(b_params_before[n], p)
        for n, p in black_model.named_parameters()
    )
    assert white_changed, "White model params unchanged after all-done training"
    assert black_changed, "Black model params unchanged after all-done training"
