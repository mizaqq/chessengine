import torch.multiprocessing as mp
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


def worker(remote, parent_remote, start_sampler=None, max_plies=None, game_start_sampler=None):
    """`start_sampler` present means this board is a puzzle board; `game_start_sampler`
    present means a mid-game board (full games from sampled FENs); `max_plies`
    caps game boards (ended as a draw)."""
    parent_remote.close()
    env = OpenSpielEnv()
    depth = {"value": 0}

    def reset_env():
        if start_sampler is not None:
            p = start_sampler.sample()
            depth["value"] = p.mate_in
            env.reset(p.fen, puzzle_moves=p.mate_in, key_moves=p.key_moves)
        elif game_start_sampler is not None:
            env.reset(game_start_sampler.sample(), max_plies=max_plies)
        else:
            env.reset(max_plies=max_plies)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "close":
                break
            elif cmd == "reset":
                reset_env()
                obs = env.state()
                legal = env.get_legal_actions().numpy()
                cp = env.get_current_player()
                remote.send((obs, _material_np(obs), legal, cp))
            elif cmd == "step":
                action = data
                env.step([action])

                obs = env.state()
                done = env.is_done()

                terminal_obs = None
                game_result = None
                ended_depth = depth["value"]
                if done:
                    terminal_obs = obs.copy()
                    game_result = env.game_result()
                    reset_env()
                    obs = env.state()

                legal = env.get_legal_actions().numpy()
                cp = env.get_current_player()
                remote.send((obs, _material_np(obs), done, terminal_obs, game_result, legal, cp, ended_depth))
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        remote.close()


class OpenSpielAsyncVectorEnv:
    """Async vectorized OpenSpiel environment using multiprocessing with EnvStep API."""

    def __init__(
        self,
        num_envs: int,
        start_sampler: StartPositionSampler | None = None,
        num_puzzle_envs: int = 0,
        max_plies: int | None = None,
        game_start_sampler=None,
        num_midgame_envs: int = 0,
    ):
        self.num_envs = num_envs
        if num_puzzle_envs > 0 and start_sampler is None:
            raise ValueError("num_puzzle_envs > 0 requires a start_sampler")
        if not 0 <= num_puzzle_envs <= num_envs:
            raise ValueError(f"num_puzzle_envs must be in [0, {num_envs}], got {num_puzzle_envs}")
        self.num_puzzle_envs = num_puzzle_envs
        if num_midgame_envs > 0 and game_start_sampler is None:
            raise ValueError("num_midgame_envs > 0 requires a game_start_sampler")
        if num_puzzle_envs + num_midgame_envs > num_envs:
            raise ValueError("num_puzzle_envs + num_midgame_envs must not exceed num_envs")
        self.num_midgame_envs = num_midgame_envs
        self.ctx = mp.get_context("fork")
        self.remotes, self.work_remotes = zip(
            *[self.ctx.Pipe() for _ in range(num_envs)]
        )
        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            # Each worker owns a sampler with its own seed so auto-reset needs no
            # round trip to the parent and runs are reproducible.
            sampler = start_sampler.with_seed(start_sampler.seed + i) if i < num_puzzle_envs else None
            midgame = (game_start_sampler.with_seed(game_start_sampler.seed + i)
                       if num_puzzle_envs <= i < num_puzzle_envs + num_midgame_envs else None)
            p = self.ctx.Process(target=worker, args=(work_remote, remote, sampler, max_plies, midgame))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self) -> EnvStep:
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, materials, legal_actions, current_players = zip(*results)
        return EnvStep(
            obs=torch.tensor(np.stack(obs)).float(),
            legal_actions_mask=torch.tensor(np.stack(legal_actions)).float(),
            material=torch.tensor(materials).float(),
            done=torch.zeros(self.num_envs, dtype=torch.bool),
            current_player=torch.tensor(current_players, dtype=torch.long),
            info={},
        )

    def step(self, actions) -> EnvStep:
        if isinstance(actions, torch.Tensor):
            actions = actions.tolist()
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

        results = [remote.recv() for remote in self.remotes]
        obs_list, materials, dones, terminal_obs_list, game_result_list, legal_list, cp_list, depth_list = (
            zip(*results)
        )

        info = {}
        terminal_observations = {}
        game_results = {}
        for i, (t_obs, g_result) in enumerate(
            zip(terminal_obs_list, game_result_list)
        ):
            if t_obs is not None:
                terminal_observations[i] = t_obs
                game_results[i] = g_result
        if terminal_observations:
            info["terminal_observations"] = terminal_observations
            info["game_results"] = game_results
            info["puzzle_boards"] = {i: i < self.num_puzzle_envs for i in game_results}
            info["puzzle_depth"] = {i: depth_list[i] for i in game_results if i < self.num_puzzle_envs}

        return EnvStep(
            obs=torch.tensor(np.stack(obs_list)).float(),
            legal_actions_mask=torch.tensor(np.stack(legal_list)).float(),
            material=torch.tensor(materials).float(),
            done=torch.tensor(dones).bool(),
            current_player=torch.tensor(cp_list, dtype=torch.long),
            info=info,
        )

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
