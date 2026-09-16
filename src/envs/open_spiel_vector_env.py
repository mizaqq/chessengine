import torch
import numpy as np
from src.core.types import EnvStep
from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.start_positions import StartPositionSampler

PIECE_VALUES = np.array([0, 9, 5, 3, 3, 1], dtype=np.float32)


def _material_np(state: np.ndarray) -> float:
    """White-view material balance of a position (queen 9, rook 5, bishop 3, knight 3, pawn 1)."""
    piece_counts = state[:12].sum(axis=(1, 2))
    white_scores = (piece_counts[0::2] * PIECE_VALUES).sum()
    black_scores = (piece_counts[1::2] * PIECE_VALUES).sum()
    return float(white_scores - black_scores)


class OpenSpielVectorEnv:
    """Vectorized OpenSpiel environment with auto-reset; reports material of each position."""

    def __init__(
        self,
        num_envs: int,
        start_sampler: StartPositionSampler | None = None,
        num_puzzle_envs: int = 0,
        max_plies: int | None = None,
    ):
        self.num_envs = num_envs
        self.envs = [OpenSpielEnv() for _ in range(num_envs)]
        if num_puzzle_envs > 0 and start_sampler is None:
            raise ValueError("num_puzzle_envs > 0 requires a start_sampler")
        if not 0 <= num_puzzle_envs <= num_envs:
            raise ValueError(f"num_puzzle_envs must be in [0, {num_envs}], got {num_puzzle_envs}")
        # Boards [0, num_puzzle_envs) play one-move puzzle episodes; the rest play games.
        self.start_sampler = start_sampler
        self.num_puzzle_envs = num_puzzle_envs
        self.max_plies = max_plies          # game boards end as a draw at this ply count
        self.puzzle_depth: dict[int, int] = {}

    def is_puzzle_env(self, i: int) -> bool:
        return i < self.num_puzzle_envs

    def _reset_env(self, i: int) -> None:
        if self.is_puzzle_env(i):
            p = self.start_sampler.sample()
            self.puzzle_depth[i] = p.mate_in
            self.envs[i].reset(p.fen, puzzle_moves=p.mate_in, key_moves=p.key_moves)
        else:
            self.envs[i].reset(max_plies=self.max_plies)

    def reset(self) -> EnvStep:
        for i in range(self.num_envs):
            self._reset_env(i)

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
        puzzle_boards: dict[int, bool] = {}
        puzzle_depth: dict[int, int] = {}

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if not env.is_done():
                env.step(int(action))

            if env.is_done():
                done[i] = True
                terminal_observations[i] = env.state().copy()
                game_results[i] = env.game_result()
                puzzle_boards[i] = self.is_puzzle_env(i)
                if puzzle_boards[i]:
                    puzzle_depth[i] = self.puzzle_depth[i]
                self._reset_env(i)

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
            info["puzzle_boards"] = puzzle_boards
            info["puzzle_depth"] = puzzle_depth

        return EnvStep(
            obs=obs,
            legal_actions_mask=legal_actions_mask,
            material=material,
            done=done,
            current_player=current_player,
            info=info,
        )
