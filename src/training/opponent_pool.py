"""Frozen-opponent pool for game boards (Bansal et al. 2017, ICLR 2018, §4.2).

Self-play against an identical copy never charges for drifting away from strong
play, because the only opponent drifted the same way (checkpoint win matrix,
2026-09-17: the supervised prior beat every RL checkpoint 88-9). The pool holds the
prior plus the learner's last `size` snapshots taken every `snapshot_every` updates;
pool boards draw one member and a learner colour per game.
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


class OpponentPool:
    def __init__(self, prior_model: Optional[torch.nn.Module], size: int = 4, snapshot_every: int = 50, seed: int = 0):
        self.prior = freeze(prior_model) if prior_model is not None else None
        self.size = int(size)
        self.snapshot_every = int(snapshot_every)
        self._snaps: deque = deque(maxlen=max(self.size, 0) or None)
        self._rng = random.Random(seed)
        if self.size == 0:
            self._snaps = deque(maxlen=0)

    @property
    def members(self) -> List[Tuple[str, torch.nn.Module]]:
        out = [("prior", self.prior)] if self.prior is not None else []
        return out + list(self._snaps)

    def maybe_snapshot(self, model: torch.nn.Module, update: int) -> bool:
        """Push a frozen copy of `model` when `update` is a multiple of `snapshot_every`."""
        if self.size <= 0 or self.snapshot_every <= 0 or update % self.snapshot_every != 0:
            return False
        self._snaps.append((f"snap_{update}", freeze(model)))
        return True

    def sample(self) -> Tuple[str, torch.nn.Module]:
        members = self.members
        if not members:
            raise ValueError("opponent pool is empty (no prior and no snapshot yet)")
        return self._rng.choice(members)


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
