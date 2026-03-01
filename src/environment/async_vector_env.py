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
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        remote.close()


class AsyncVectorEnv(VectorEnv):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[self.ctx.Pipe() for _ in range(num_envs)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = self.ctx.Process(target=worker, args=(work_remote, remote))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self):
        pass

    def step(self, actions):
        pass

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
