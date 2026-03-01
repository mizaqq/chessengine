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
    """Vectorized OpenSpiel environment with auto-reset and piece-difference reward."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [OpenSpielEnv() for _ in range(num_envs)]
        self._previous_states: list[np.ndarray] = []

    def reset(self) -> EnvStep:
        for env in self.envs:
            env.reset()

        states = [env.state() for env in self.envs]
        self._previous_states = [s.copy() for s in states]

        obs = torch.tensor(np.array(states, dtype=np.float32))
        legal_actions_mask = torch.stack([env.get_legal_actions() for env in self.envs])
        reward = torch.zeros(self.num_envs)
        done = torch.tensor([False] * self.num_envs, dtype=torch.bool)

        return EnvStep(
            obs=obs,
            legal_actions_mask=legal_actions_mask,
            reward=reward,
            done=done,
            info={},
        )

    def step(self, actions) -> EnvStep:
        rewards = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        terminal_observations: dict[int, np.ndarray] = {}
        game_results: dict[int, str] = {}

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if not env.is_done():
                env.step(int(action))

            current_state = env.state()
            rewards[i] = _piece_difference_np(current_state) - _piece_difference_np(
                self._previous_states[i]
            )

            if env.is_done():
                done[i] = True
                terminal_observations[i] = current_state.copy()
                game_results[i] = env.game_result()
                env.reset()
                current_state = env.state()

            self._previous_states[i] = current_state.copy()

        states = [env.state() for env in self.envs]
        obs = torch.tensor(np.array(states, dtype=np.float32))
        legal_actions_mask = torch.stack([env.get_legal_actions() for env in self.envs])

        info: dict = {}
        if terminal_observations:
            info["terminal_observations"] = terminal_observations
            info["game_results"] = game_results

        return EnvStep(
            obs=obs,
            legal_actions_mask=legal_actions_mask,
            reward=rewards,
            done=done,
            info=info,
        )
