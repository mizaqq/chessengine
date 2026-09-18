"""Self-imitation learning (Oh, Guo, Singh & Lee, ICML 2018) for the shared network.

Every learner ply is held per board until its episode ends, then stored with the
Monte Carlo return R from the mover's side. After each PPO update, M minibatches are
drawn with probability proportional to the priority (R - V)+ and trained with
    L = w * [ -log pi(a|s) (R - V)+  +  beta * 1/2 ||(R - V)+||^2 ]      (paper eq. 1-3)
so only plies whose past return beat the current value estimate get gradient, and a
rare win is learned from many times. No importance ratio (paper §4: the objective is a
lower bound on the optimal soft Q-value for any behaviour that did better than the
learner). Paper's PPO table: M 10, batch 512, w 0.1, beta 0.01, buffer 50k for 20k
on-policy samples per iteration; ours has ~1k per update, so M 4 x 256.

Discounting follows the value targets of the PPO update (returns.py): gamma is applied
once per own ply, and the ply that ends the game gets gamma * terminal, so R sits on
the same scale as V.
"""
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.distributions import Categorical

WHITE, BLACK = 1, 0
NO_PROGRESS_PLANE = 15
OBS_BITS, MASK_BITS = 20 * 64, 4674


def _pack_obs(obs: torch.Tensor):
    arr = obs.numpy()
    nop = float(arr[NO_PROGRESS_PLANE, 0, 0])
    planes = arr.copy(); planes[NO_PROGRESS_PLANE] = 0
    return np.packbits(planes.astype(bool).ravel()), nop


def _unpack_obs(bits: np.ndarray, nop: np.ndarray) -> torch.Tensor:
    obs = torch.tensor(np.unpackbits(bits, axis=1)[:, :OBS_BITS].reshape(-1, 20, 8, 8), dtype=torch.float32)
    obs[:, NO_PROGRESS_PLANE] = torch.tensor(nop, dtype=torch.float32)[:, None, None]
    return obs


def _unpack_mask(bits: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.unpackbits(bits, axis=1)[:, :MASK_BITS], dtype=torch.float32)


class ReplayBuffer:
    """Ring of packed plies with a priority per ply."""

    def __init__(self, capacity: int, alpha: float = 1.0, beta: float = 0.1, seed: int = 0):
        self.capacity = int(capacity)
        self.alpha, self.beta = float(alpha), float(beta)
        self.obs = np.zeros((self.capacity, OBS_BITS // 8), np.uint8)
        self.nop = np.zeros(self.capacity, np.float32)
        self.mask = np.zeros((self.capacity, (MASK_BITS + 7) // 8), np.uint8)
        self.action = np.zeros(self.capacity, np.int64)
        self.ret = np.zeros(self.capacity, np.float32)
        self.priority = np.zeros(self.capacity, np.float32)
        self.size = self.next = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, obs: torch.Tensor, mask: torch.Tensor, action: int, ret: float, value: float):
        i = self.next
        self.obs[i], self.nop[i] = _pack_obs(obs)
        self.mask[i] = np.packbits(mask.numpy().astype(bool))
        self.action[i], self.ret[i] = int(action), float(ret)
        self.priority[i] = max(float(ret) - float(value), 0.0)
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        """(indices, bias-correction weights) or None when no ply has positive priority."""
        if self.size == 0:
            return None
        p = self.priority[: self.size] ** self.alpha
        total = p.sum()
        if total <= 0:
            return None
        p = p / total
        idx = self._rng.choice(self.size, size=min(batch, int((p > 0).sum())), replace=True, p=p)
        w = (self.size * p[idx]) ** (-self.beta)
        return idx, torch.tensor(w / w.max(), dtype=torch.float32)

    def get(self, idx):
        return (_unpack_obs(self.obs[idx], self.nop[idx]), _unpack_mask(self.mask[idx]),
                torch.tensor(self.action[idx]), torch.tensor(self.ret[idx]))

    def positive_share(self) -> float:
        """Share of stored plies whose return still beats the value estimate they were
        last compared with (the informative SIL dial; the share among drawn plies is ~1
        by construction of the prioritised draw)."""
        return float((self.priority[: self.size] > 0).mean()) if self.size else 0.0

    def update_priority(self, idx, adv: torch.Tensor):
        self.priority[idx] = adv.clamp(min=0).numpy()


class EpisodeAccumulator:
    """Holds each board's learner plies until the episode ends, then pushes them with
    their returns. Returns: gamma once per own ply, the game-ending ply gets
    gamma * terminal (same convention as the value targets)."""

    def __init__(self, num_envs: int, gamma: float, buffer: ReplayBuffer):
        self.gamma = float(gamma)
        self.buffer = buffer
        self.pending: List[list] = [[] for _ in range(num_envs)]

    def add_step(self, step) -> int:
        """Feed one rollout step (all boards). Returns the number of plies pushed."""
        pushed = 0
        learner = step.learner if step.learner is not None else torch.ones(len(step.player), dtype=torch.bool)
        for i in range(len(step.player)):
            if bool(learner[i]):
                self.pending[i].append((step.obs[i], step.legal_mask[i], int(step.action[i]),
                                        int(step.player[i]), float(step.old_value[i])))
            if bool(step.done[i]):
                pushed += self._finish(i, float(step.terminal_r_white[i]), float(step.terminal_r_black[i]))
        return pushed

    def _finish(self, i: int, tr_white: float, tr_black: float) -> int:
        plies = self.pending[i]
        self.pending[i] = []
        remaining = {WHITE: 0, BLACK: 0}
        rets = [0.0] * len(plies)
        for k in range(len(plies) - 1, -1, -1):
            side = plies[k][3]
            remaining[side] += 1
            terminal = tr_white if side == WHITE else tr_black
            rets[k] = terminal * self.gamma ** remaining[side]
        for (obs, mask, action, _side, value), r in zip(plies, rets):
            self.buffer.add(obs, mask, action, r, value)
        return len(plies)


def sil_loss(model, obs, mask, action, ret, weights, value_weight: float):
    """Paper eq. 2-3 with bias-correction weights. Returns (loss, stats)."""
    probs, value = model(obs, mask)
    value = value.squeeze(-1)
    adv = (ret - value.detach()).clamp(min=0)
    logp = Categorical(probs=probs).log_prob(action)
    policy = (-logp * adv * weights).mean()
    value_term = 0.5 * (((ret - value).clamp(min=0)) ** 2 * weights).mean()
    loss = policy + value_weight * value_term
    stats = {"sil_policy_loss": policy.item(), "sil_value_loss": value_term.item(),
             "sil_valid_share": (adv > 0).float().mean().item()}
    return loss, adv.detach(), stats


def sil_update(model, optimizer, buffer: ReplayBuffer, *, updates: int, batch: int, loss_weight: float,
               value_weight: float, grad_clip: float) -> Optional[Dict[str, float]]:
    """M minibatches of self-imitation after the on-policy update. None if the buffer
    has nothing with positive priority."""
    sums: Dict[str, float] = {}
    steps = 0
    for _ in range(updates):
        drawn = buffer.sample(batch)
        if drawn is None:
            break
        idx, w = drawn
        obs, mask, action, ret = buffer.get(idx)
        loss, adv, stats = sil_loss(model, obs, mask, action, ret, w, value_weight)
        optimizer.zero_grad()
        (loss_weight * loss).backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=grad_clip)
        optimizer.step()
        buffer.update_priority(idx, adv)
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + v
        steps += 1
    if steps == 0:
        return None
    out = {k: v / steps for k, v in sums.items()}
    out["sil_buffer_size"] = len(buffer)
    out["sil_positive_share"] = buffer.positive_share()
    return out
