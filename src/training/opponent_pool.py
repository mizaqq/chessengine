"""Frozen-opponent pool for game boards (Bansal et al. 2017, ICLR 2018, §4.2).

Self-play against an identical copy never charges for drifting away from strong
play, because the only opponent drifted the same way (checkpoint win matrix,
2026-09-17: the supervised prior beat every RL checkpoint 88-9). The pool holds the
prior plus the learner's last `size` snapshots taken every `snapshot_every` updates;
pool boards draw one member and a learner colour per game.

Sampling (change prior-anchored-selfplay, 2026-09-22): `uniform` draws every member
equally (fictitious self-play over a short window, the weakest line in AlphaStar's
ablation: FSP 1143 < pFSP 1273 < SP 1519 < pFSP+SP 1540). `pfsp` is AlphaStar's
prioritised fictitious self-play with f_hard: member m is drawn with weight
`(1 - w_m)^power + min_weight`, w_m the learner's smoothed win rate against m (EMA 0.1,
start 0.5), so the games go to the opponents the learner currently loses to, and a
beaten snapshot the learner later starts losing to again comes back ("forgotten
players"). The floor keeps every win rate fresh; f_hard(1) = 0 alone never re-probes.
"""
import copy
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import torch

WHITE, BLACK = 1, 0


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    snap = copy.deepcopy(model).eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


EMA_WEIGHT = 0.1


class OpponentPool:
    def __init__(self, prior_model: Optional[torch.nn.Module], size: int = 4, snapshot_every: int = 50, seed: int = 0,
                 sampling: str = "uniform", power: float = 2.0, min_weight: float = 0.05):
        self.prior = freeze(prior_model) if prior_model is not None else None
        self.size = int(size)
        self.snapshot_every = int(snapshot_every)
        self._snaps: deque = deque(maxlen=max(self.size, 0) or None)
        self._rng = random.Random(seed)
        if self.size == 0:
            self._snaps = deque(maxlen=0)
        if sampling not in ("uniform", "pfsp"):
            raise ValueError(f"opponent_pool_sampling must be 'uniform' or 'pfsp', got {sampling!r}")
        self.sampling = sampling
        self.power = float(power)
        self.min_weight = float(min_weight)
        if self.power < 0 or self.min_weight < 0:
            raise ValueError("pfsp_power and pfsp_min_weight must be >= 0")
        self.win_rate: Dict[str, float] = {}       # member name -> learner's smoothed score against it
        if self.prior is not None:
            self.win_rate["prior"] = 0.5

    @property
    def members(self) -> List[Tuple[str, torch.nn.Module]]:
        out = [("prior", self.prior)] if self.prior is not None else []
        return out + list(self._snaps)

    def maybe_snapshot(self, model: torch.nn.Module, update: int) -> bool:
        """Push a frozen copy of `model` when `update` is a multiple of `snapshot_every`."""
        if self.size <= 0 or self.snapshot_every <= 0 or update % self.snapshot_every != 0:
            return False
        name = f"snap_{update}"
        self._snaps.append((name, freeze(model)))     # same device as the learner
        self.win_rate[name] = 0.5
        self.win_rate = {n: w for n, w in self.win_rate.items() if n in {m for m, _ in self.members}}
        return True

    def record(self, name: str, score: float) -> None:
        """A finished pool game: learner's score (1 / 0.5 / 0) against member `name`."""
        if name in self.win_rate:
            self.win_rate[name] = (1 - EMA_WEIGHT) * self.win_rate[name] + EMA_WEIGHT * float(score)

    def weights(self) -> Dict[str, float]:
        """Normalised draw probability per member under the current sampling rule."""
        members = self.members
        if not members:
            return {}
        if self.sampling == "uniform":
            return {n: 1.0 / len(members) for n, _ in members}
        raw = {n: (1.0 - self.win_rate.get(n, 0.5)) ** self.power + self.min_weight for n, _ in members}
        total = sum(raw.values())
        if total <= 0:
            return {n: 1.0 / len(members) for n, _ in members}
        return {n: v / total for n, v in raw.items()}

    def sample(self) -> Tuple[str, torch.nn.Module]:
        members = self.members
        if not members:
            raise ValueError("opponent pool is empty (no prior and no snapshot yet)")
        if self.sampling == "uniform":
            return self._rng.choice(members)
        w = self.weights()
        return self._rng.choices(members, weights=[w[n] for n, _ in members], k=1)[0]


class PoolBoards:
    """Per-board (opponent name, opponent network, learner colour) for the pool boards."""

    def __init__(self, board_ids: Sequence[int], pool: OpponentPool, seed: int = 0):
        self.board_ids = list(board_ids)
        self.pool = pool
        self._rng = random.Random(seed)
        self._state: Dict[int, Tuple[str, torch.nn.Module, int]] = {}
        self.redraw(self.board_ids)

    def redraw(self, done_ids) -> None:
        for i in done_ids:
            if i in self.board_ids:
                name, model = self.pool.sample()
                self._state[i] = (name, model, self._rng.choice((WHITE, BLACK)))

    def assignment(self, i: int) -> Tuple[str, int]:
        name, _, colour = self._state[i]
        return name, colour

    def learner_colour(self, i: int) -> int:
        return self._state[i][2]

    def learner_mask(self, players: torch.Tensor) -> torch.Tensor:
        """True where the side to move is the learner (all non-pool boards are True)."""
        mask = torch.ones(len(players), dtype=torch.bool)
        for i in self.board_ids:
            mask[i] = int(players[i]) == self._state[i][2]
        return mask

    def opponent_groups(self, players: torch.Tensor) -> Dict[torch.nn.Module, List[int]]:
        """Boards where the opponent is to move, grouped by opponent network."""
        groups: Dict[torch.nn.Module, List[int]] = {}
        for i in self.board_ids:
            name, model, colour = self._state[i]
            if int(players[i]) != colour:
                groups.setdefault(model, []).append(i)
        return groups

    def opponent_name(self, i: int) -> str:
        return self._state[i][0]
