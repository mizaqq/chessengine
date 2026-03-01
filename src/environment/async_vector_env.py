import torch.multiprocessing as mp
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from src.core.interfaces import VectorEnv
from src.envs.open_spiel_env import OpenSpielEnv


PIECE_VALUES = np.array([0, 9, 5, 3, 3, 1], dtype=np.float32)


def _piece_difference_np(state: np.ndarray) -> float:
    """Compute white_score - black_score from a (20, 8, 8) observation using numpy."""
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

                info = {}
                if done:
                    info["terminal_observation"] = obs.copy()
                    env.reset()
                    obs = env.state()

                previous_state = obs
                legal = env.get_legal_actions().numpy()
                remote.send((obs, reward, done, info, legal))
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        remote.close()


class AsyncVectorEnv(VectorEnv):
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

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, legal_actions = zip(*results)
        return torch.tensor(np.stack(obs)).float(), torch.tensor(np.stack(legal_actions)).float()

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
            torch.tensor(np.stack(legal_actions)).float(),
        )

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
