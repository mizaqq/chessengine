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
