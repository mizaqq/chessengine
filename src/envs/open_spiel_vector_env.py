import torch
import numpy as np
from src.core.types import EnvStep
from src.envs.open_spiel_env import OpenSpielEnv

PIECE_VALUES = np.array([0, 9, 5, 3, 3, 1], dtype=np.float32)


def _material_np(state: np.ndarray) -> float:
    """White-view material balance of a position (queen 9, rook 5, bishop 3, knight 3, pawn 1)."""
    piece_counts = state[:12].sum(axis=(1, 2))
    white_scores = (piece_counts[0::2] * PIECE_VALUES).sum()
    black_scores = (piece_counts[1::2] * PIECE_VALUES).sum()
    return float(white_scores - black_scores)


class OpenSpielVectorEnv:
    """Vectorized OpenSpiel environment with auto-reset; reports material of each position."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [OpenSpielEnv() for _ in range(num_envs)]

    def reset(self) -> EnvStep:
        for env in self.envs:
            env.reset()

        states = [env.state() for env in self.envs]

        obs = torch.tensor(np.array(states, dtype=np.float32))
        legal_actions_mask = torch.stack([env.get_legal_actions() for env in self.envs])
        material = torch.tensor([_material_np(s) for s in states])
        done = torch.tensor([False] * self.num_envs, dtype=torch.bool)
        current_player = torch.tensor(
            [env.get_current_player() for env in self.envs], dtype=torch.long
        )

        return EnvStep(
            obs=obs,
            legal_actions_mask=legal_actions_mask,
            material=material,
            done=done,
            current_player=current_player,
            info={},
        )

    def step(self, actions) -> EnvStep:
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        terminal_observations: dict[int, np.ndarray] = {}
        game_results: dict[int, str] = {}

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if not env.is_done():
                env.step(int(action))

            if env.is_done():
                done[i] = True
                terminal_observations[i] = env.state().copy()
                game_results[i] = env.game_result()
                env.reset()

        states = [env.state() for env in self.envs]
        obs = torch.tensor(np.array(states, dtype=np.float32))
        material = torch.tensor([_material_np(s) for s in states])
        legal_actions_mask = torch.stack([env.get_legal_actions() for env in self.envs])
        current_player = torch.tensor(
            [env.get_current_player() for env in self.envs], dtype=torch.long
        )

        info: dict = {}
        if terminal_observations:
            info["terminal_observations"] = terminal_observations
            info["game_results"] = game_results

        return EnvStep(
            obs=obs,
            legal_actions_mask=legal_actions_mask,
            material=material,
            done=done,
            current_player=current_player,
            info=info,
        )
