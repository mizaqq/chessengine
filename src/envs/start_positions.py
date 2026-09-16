"""Start positions for training games: puzzle loading and the mixing sampler.

The first `num_puzzle_envs` boards of a vectorized env are puzzle boards: every
episode starts from a training puzzle drawn by a `StartPositionSampler` and ends
after the mover's first move. The other boards play normal games from the opening.
"""
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    fen: str                 # position where the side to move has a mate in one
    mating_moves: List[str]  # UCI
    rating: int


def load_puzzles(path) -> List[Puzzle]:
    with open(Path(path), newline="") as fh:
        return [
            Puzzle(
                puzzle_id=row["puzzle_id"],
                fen=row["fen"],
                mating_moves=row["mating_moves"].split(),
                rating=int(row["rating"]),
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

    def sample(self) -> str:
        return self._rng.choice(self.puzzles).fen

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

    def sample(self) -> str:
        puzzles, = self._rng.choices([p for p, _ in self.sources], weights=self.weights, k=1)
        return self._rng.choice(puzzles).fen

    def with_seed(self, seed: int) -> "MixedSampler":
        return MixedSampler(self.sources, seed)
