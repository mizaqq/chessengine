import torch
import numpy as np
from src.core.types import EnvStep
from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.start_positions import StartPositionSampler

WHITE = 1  # OpenSpiel convention

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
        game_start_sampler=None,
        num_midgame_envs: int = 0,
        finish_sampler=None,
        num_finish_envs: int = 0,
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
        # Layout after the puzzle boards: finishing boards [P, P+F) start k plies
        # before a human mate under a per-board cap (reverse curriculum, Florensa
        # et al. 2017); mid-game boards [P+F, P+F+M) start full games from sampled
        # mid-game FENs; the rest start from the opening.
        if num_finish_envs > 0 and finish_sampler is None:
            raise ValueError("num_finish_envs > 0 requires a finish_sampler")
        if num_midgame_envs > 0 and game_start_sampler is None:
            raise ValueError("num_midgame_envs > 0 requires a game_start_sampler")
        if num_puzzle_envs + num_finish_envs + num_midgame_envs > num_envs:
            raise ValueError("num_puzzle_envs + num_finish_envs + num_midgame_envs must not exceed num_envs")
        self.finish_sampler = finish_sampler
        self.num_finish_envs = num_finish_envs
        self.finish_start: dict[int, object] = {}
        self.game_start_sampler = game_start_sampler
        self.num_midgame_envs = num_midgame_envs

    def set_layout(self, num_finish_envs: int, num_midgame_envs: int) -> None:
        """Change the board roles after the puzzle boards; each board takes its new
        role when it next resets, so no running game is cut. Boards beyond the
        puzzle, finishing and mid-game boards start from the opening."""
        if num_finish_envs > 0 and self.finish_sampler is None:
            raise ValueError("num_finish_envs > 0 requires a finish_sampler")
        if num_midgame_envs > 0 and self.game_start_sampler is None:
            raise ValueError("num_midgame_envs > 0 requires a game_start_sampler")
        if num_finish_envs < 0 or num_midgame_envs < 0 or \
                self.num_puzzle_envs + num_finish_envs + num_midgame_envs > self.num_envs:
            raise ValueError("num_puzzle_envs + num_finish_envs + num_midgame_envs must not exceed num_envs")
        self.num_finish_envs = num_finish_envs
        self.num_midgame_envs = num_midgame_envs

    def demo_actions(self, boards=None):
        """Per-board demonstrator action sets (label-guided exploration); `boards`
        restricts the rules-engine work to those indices, others get empty sets."""
        return [self.envs[i].demo_actions() if (boards is None or boards[i]) else set()
                for i in range(self.num_envs)]

    def is_puzzle_env(self, i: int) -> bool:
        return i < self.num_puzzle_envs

    def is_finish_env(self, i: int) -> bool:
        return self.num_puzzle_envs <= i < self.num_puzzle_envs + self.num_finish_envs

    def is_midgame_env(self, i: int) -> bool:
        lo = self.num_puzzle_envs + self.num_finish_envs
        return lo <= i < lo + self.num_midgame_envs

    def _reset_env(self, i: int) -> None:
        if self.is_puzzle_env(i):
            p = self.start_sampler.sample()
            self.puzzle_depth[i] = p.mate_in
            self.envs[i].reset(p.fen, puzzle_moves=p.mate_in, key_moves=p.key_moves,
                               solution=getattr(p, "solution", ()))
        elif self.is_finish_env(i):
            s = self.finish_sampler.sample()
            self.finish_start[i] = s
            self.envs[i].reset(s.fen, max_plies=s.cap)
            self.envs[i].set_demo_line(getattr(s, "line", None))
        elif self.is_midgame_env(i):
            self.envs[i].reset(self.game_start_sampler.sample(), max_plies=self.max_plies)
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
        finish_boards: dict[int, bool] = {}
        finish_depth: dict[int, int] = {}
        finish_success: dict[int, bool] = {}
        finish_config: dict[int, str] = {}

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
                finish_boards[i] = i in self.finish_start
                if finish_boards[i]:
                    start = self.finish_start[i]
                    won = game_results[i] == ("white_win" if start.winner == WHITE else "black_win")
                    finish_depth[i] = start.depth
                    finish_success[i] = won
                    finish_config[i] = getattr(start, "config", "")
                    if not getattr(start, "config", ""):
                        # Labelled (technique) starts are reported by the training loop,
                        # which knows whether the demonstrator acted in the episode.
                        self.finish_sampler.report(start.depth, won)
                    del self.finish_start[i]
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
            info["finish_boards"] = finish_boards
            info["finish_depth"] = finish_depth
            info["finish_success"] = finish_success
            info["finish_config"] = finish_config

        return EnvStep(
            obs=obs,
            legal_actions_mask=legal_actions_mask,
            material=material,
            done=done,
            current_player=current_player,
            info=info,
        )
