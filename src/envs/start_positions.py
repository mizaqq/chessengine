"""Start positions for training games: puzzle loading and the mixing sampler.

A `StartPositionSampler` decides, on every reset, whether a board starts from the
opening (returns None) or from a training puzzle (returns its FEN). The fraction is
fixed; the sampler is deterministic for a given seed.
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
    """Return a puzzle FEN with probability `fraction`, else None (opening)."""

    def __init__(self, puzzles: List[Puzzle], fraction: float, seed: int = 0):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"puzzle_fraction must be in [0, 1], got {fraction}")
        if fraction > 0 and not puzzles:
            raise ValueError("puzzle_fraction > 0 requires a non-empty puzzle set")
        self.puzzles = list(puzzles)
        self.fraction = fraction
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(self) -> Optional[str]:
        if self.fraction <= 0.0:
            return None
        if self.fraction < 1.0 and self._rng.random() >= self.fraction:
            return None
        return self._rng.choice(self.puzzles).fen

    def with_seed(self, seed: int) -> "StartPositionSampler":
        """Copy with a different seed (one per worker in the async env)."""
        return StartPositionSampler(self.puzzles, self.fraction, seed)
