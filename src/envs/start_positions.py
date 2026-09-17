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


def load_fens(path) -> List[str]:
    """CSV with a `fen` column (scripts/prepare_games.py writes midgame_starts.csv)."""
    with open(Path(path), newline="") as fh:
        return [row["fen"] for row in csv.DictReader(fh)]


class FenSampler:
    """Deterministic stream of start FENs for mid-game boards (full games, not
    puzzles); same interface as the puzzle samplers."""

    def __init__(self, fens: List[str], seed: int = 0):
        if not fens:
            raise ValueError("mid-game boards require a non-empty FEN list")
        self.fens = list(fens)
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(self) -> str:
        return self._rng.choice(self.fens)

    def with_seed(self, seed: int) -> "FenSampler":
        return FenSampler(self.fens, seed)


# --- Finishing curriculum (reverse curriculum from human checkmates) -----------
# Florensa et al. 2017: start near the goal, expand the start distribution
# backwards as the success rate allows. Salimans & Chen 2018: exploration cost
# grows with the distance between start state and reward.

WHITE = 1   # OpenSpiel convention
BLACK = 0


@dataclass(frozen=True)
class FinishRecord:
    game_id: str
    anchor_fen: str          # position at most 40 plies before the end
    moves: List[str]         # UCI from the anchor up to and including the mating move
    winner: int              # WHITE / BLACK: the side that delivered mate

    def max_depth(self) -> int:
        """Largest even rewind (plies before the mating move) this record supports."""
        n = len(self.moves) - 1
        return n if n % 2 == 0 else n - 1

    def rewind(self, depth: int) -> str:
        """FEN `depth` plies before the mating move; the winner is to move for even depth."""
        import chess
        board = chess.Board(self.anchor_fen)
        for uci in self.moves[: len(self.moves) - 1 - depth]:
            board.push_uci(uci)
        return board.fen()


def load_finishes(path) -> List[FinishRecord]:
    """CSV `game_id,anchor_fen,moves,winner` (winner `white`/`black`), written by
    scripts/prepare_finishes.py."""
    with open(Path(path), newline="") as fh:
        return [
            FinishRecord(row["game_id"], row["anchor_fen"], row["moves"].split(),
                         WHITE if row["winner"] == "white" else BLACK)
            for row in csv.DictReader(fh)
        ]


@dataclass(frozen=True)
class FinishStart:
    fen: str
    depth: int
    cap: int      # plies allowed from `fen` before the game is a draw
    winner: int


class FinishCurriculum:
    """Sampler for finishing boards with an adaptive depth.

    `sample()` draws a record and an even depth uniformly from [0, current_depth]
    (clipped to what the record supports) and returns the rewound start with its
    cap `depth + cap_margin`. `report(depth, success)` feeds the rolling window;
    when at least `window` episodes were reported since the last rise and their
    success rate is >= `advance_rate`, `current_depth` rises by 2 up to
    `depth_max`. It never decreases."""

    def __init__(self, records: List[FinishRecord], depth_start: int = 2, depth_max: int = 40,
                 advance_rate: float = 0.6, window: int = 100, cap_margin: int = 20, seed: int = 0):
        if not records:
            raise ValueError("finishing boards require a non-empty record set")
        if depth_start < 0 or depth_start % 2 or depth_max < depth_start or depth_max % 2:
            raise ValueError(f"depths must be even with 0 <= depth_start <= depth_max, got {depth_start}, {depth_max}")
        if not 0.0 < advance_rate <= 1.0:
            raise ValueError(f"advance_rate must be in (0, 1], got {advance_rate}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if cap_margin < 2:
            raise ValueError(f"cap_margin must be >= 2, got {cap_margin}")
        self.records = list(records)
        self.depth_start, self.depth_max = depth_start, depth_max
        self.advance_rate, self.window, self.cap_margin = advance_rate, window, cap_margin
        self.seed = seed
        self._rng = random.Random(seed)
        self.current_depth = depth_start
        self._recent: List[bool] = []      # last `window` outcomes
        self._since_rise = 0

    def sample(self) -> FinishStart:
        rec = self._rng.choice(self.records)
        depth = 2 * self._rng.randint(0, self.current_depth // 2)
        depth = min(depth, rec.max_depth())
        return FinishStart(rec.rewind(depth), depth, depth + self.cap_margin, rec.winner)

    def report(self, depth: int, success: bool) -> None:
        self._recent.append(bool(success))
        if len(self._recent) > self.window:
            self._recent.pop(0)
        self._since_rise += 1
        if self._since_rise >= self.window and self.current_depth < self.depth_max:
            rate = sum(self._recent) / len(self._recent)
            if rate >= self.advance_rate:
                self.current_depth = min(self.current_depth + 2, self.depth_max)
                self._since_rise = 0

    def with_seed(self, seed: int) -> "FinishCurriculum":
        return FinishCurriculum(self.records, self.depth_start, self.depth_max, self.advance_rate,
                                self.window, self.cap_margin, seed)
