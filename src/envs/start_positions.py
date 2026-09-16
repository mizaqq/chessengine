"""Start positions for training games: puzzle loading and the mixing sampler.

The first `num_puzzle_envs` boards of a vectorized env are puzzle boards: every
episode starts from a training puzzle drawn by a `StartPositionSampler` and ends
after the mover's `mate_in`-th move (or at once after a wrong first move). The other boards play normal games from the opening.
"""
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    fen: str                 # position where the side to move has a forced mate in `mate_in`
    key_moves: List[str]     # UCI: all mating moves (mate_in 1) or the forcing first move(s) (mate_in 2)
    rating: int
    mate_in: int = 1

    @property
    def mating_moves(self) -> List[str]:   # legacy name
        return self.key_moves


def load_puzzles(path) -> List[Puzzle]:
    """CSV with `puzzle_id,fen,key_moves,mate_in,rating`; the older header
    `mating_moves` (no `mate_in`) loads as mate-in-one."""
    with open(Path(path), newline="") as fh:
        return [
            Puzzle(
                puzzle_id=row["puzzle_id"],
                fen=row["fen"],
                key_moves=(row.get("key_moves") or row.get("mating_moves") or "").split(),
                rating=int(row["rating"]),
                mate_in=int(row.get("mate_in") or 1),
            )
            for row in csv.DictReader(fh)
        ]


class StartPositionSampler:
    """Deterministic stream of training puzzle FENs for the puzzle boards."""

    def __init__(self, puzzles: List[Puzzle], seed: int = 0):
        if not puzzles:
            raise ValueError("puzzle boards require a non-empty puzzle set")
        self.puzzles = list(puzzles)
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(self) -> Puzzle:
        return self._rng.choice(self.puzzles)

    def with_seed(self, seed: int) -> "StartPositionSampler":
        """Copy with a different seed (one per worker in the async env)."""
        return StartPositionSampler(self.puzzles, seed)


class MixedSampler:
    """Draw from several puzzle sets with normalised weights; same interface as
    `StartPositionSampler` (`sample()`, `with_seed()`, `.seed`)."""

    def __init__(self, sources, seed: int = 0):
        sources = [(list(p), float(w)) for p, w in sources]
        if not sources:
            raise ValueError("MixedSampler needs at least one source")
        if any(w < 0 for _, w in sources):
            raise ValueError("puzzle source weights must be >= 0")
        total = sum(w for _, w in sources)
        if total <= 0:
            raise ValueError("puzzle source weights must sum to > 0")
        for puzzles, w in sources:
            if w > 0 and not puzzles:
                raise ValueError("a puzzle source with positive weight is empty")
        self.sources = sources
        self.weights = [w / total for _, w in sources]
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(self) -> Puzzle:
        puzzles, = self._rng.choices([p for p, _ in self.sources], weights=self.weights, k=1)
        return self._rng.choice(puzzles)

    def with_seed(self, seed: int) -> "MixedSampler":
        return MixedSampler(self.sources, seed)
