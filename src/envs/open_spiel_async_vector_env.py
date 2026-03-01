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
            if cmd == "close":
                break
            elif cmd == "reset":
                env.reset()
                obs = env.state()
                previous_state = obs
                legal = env.get_legal_actions().numpy()
                remote.send((obs, legal))
            elif cmd == "step":
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
    """Async vectorized OpenSpiel environment using multiprocessing with EnvStep API."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.ctx = mp.get_context("fork")
        self.remotes, self.work_remotes = zip(
            *[self.ctx.Pipe() for _ in range(num_envs)]
        )
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = self.ctx.Process(target=worker, args=(work_remote, remote))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self) -> EnvStep:
        for remote in self.remotes:
            remote.send(("reset", None))
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
            remote.send(("step", action))

        results = [remote.recv() for remote in self.remotes]
        obs_list, rewards, dones, terminal_obs_list, game_result_list, legal_list = (
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

        return EnvStep(
            obs=torch.tensor(np.stack(obs_list)).float(),
            legal_actions_mask=torch.tensor(np.stack(legal_list)).float(),
            reward=torch.tensor(rewards).float(),
            done=torch.tensor(dones).bool(),
            info=info,
        )

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
