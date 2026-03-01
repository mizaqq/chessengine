import torch.multiprocessing as mp
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from src.core.interfaces import VectorEnv
from src.envs.open_spiel_env import OpenSpielEnv


def worker(remote, parent_remote):
    parent_remote.close()
    env = OpenSpielEnv()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'close':
                break
            elif cmd == 'reset':
                env.reset()
                legal = env.get_legal_actions().numpy()
                remote.send((env.state(), legal))
            elif cmd == 'step':
                action = data
                env.step([action])

                obs = env.state()
                done = env.is_done()
                reward = 0.0

                info = {}
                if done:
                    info["terminal_observation"] = obs.copy()
                    env.reset()
                    obs = env.state()

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
