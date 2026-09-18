"""Generated technique starts: bare-king endings with a forced win by rule.

Owner 2026-09-18: "the bare king stuff should be generated" (the Lichess puzzle
database has none). A material set is the strong side's pieces beyond the king
("Q", "R", "RR", "QR"). The strong side is to move; positions with a mate in one
available, or with the strong king in check, or stalemate, are rejected. Each start
carries the set's ply cap; the board is a draw at the cap (a failure). Perfect-play
worst cases: KQ v K 10 moves, KR v K 16, KRR 7, KQR 6, so caps of 40 / 60 / 40 / 40
plies are about twice the worst case.

Start-state curriculum (owner 2026-09-18, after arms TECH/TECH2 where the held-out
technique rate never left zero; Florensa et al. 2017 without a ruler): each set has a
level, 0 = weak king in a corner with the strong king at distance 2 and the pieces
within 4, 1 = weak king on an edge with the strong king within 3, 2 = weak king one
square from an edge, 3 = unconstrained. The level rises when the last `window`
clean boards of that set (no demonstrator move in the episode) were won at
`advance_rate` or better; starts draw a level uniformly in [0, current] so easy
positions stay in the mix.
"""
import random
from collections import deque
from typing import Dict, Optional, Sequence

import chess

from src.envs.start_positions import BLACK, WHITE, FinishStart

DEFAULT_CAPS: Dict[str, int] = {"Q": 40, "R": 60, "RR": 40, "QR": 40}
MAX_LEVEL = 3
_PIECE = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}
CORNERS = [chess.A1, chess.A8, chess.H1, chess.H8]
EDGE = [sq for sq in range(64) if chess.square_rank(sq) in (0, 7) or chess.square_file(sq) in (0, 7)]
NEAR_EDGE = [sq for sq in range(64) if sq not in EDGE and
             (chess.square_rank(sq) in (1, 6) or chess.square_file(sq) in (1, 6))]


def _has_mate_in_one(board: chess.Board) -> bool:
    for move in board.legal_moves:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def _level_squares(level: int, rng: random.Random, n_pieces: int):
    """(weak king square, strong king square, piece squares) candidate for a level."""
    if level == 0:
        weak = rng.choice(CORNERS)
        kings = [sq for sq in range(64) if chess.square_distance(sq, weak) == 2]
        # Pieces within 4 of the cornered king: within 2 almost every queen position is
        # a mate in one (rejected) or an illegal check (measured 2026-09-18: 0 of 300).
        near = [sq for sq in range(64) if 1 <= chess.square_distance(sq, weak) <= 4]
    elif level == 1:
        weak = rng.choice(EDGE)
        kings = [sq for sq in range(64) if 2 <= chess.square_distance(sq, weak) <= 3]
        near = [sq for sq in range(64) if sq != weak]
    elif level == 2:
        weak = rng.choice(NEAR_EDGE)
        kings = [sq for sq in range(64) if chess.square_distance(sq, weak) >= 2]
        near = [sq for sq in range(64) if sq != weak]
    else:
        weak = rng.randrange(64)
        kings = [sq for sq in range(64) if chess.square_distance(sq, weak) >= 2]
        near = [sq for sq in range(64) if sq != weak]
    strong = rng.choice(kings)
    free = [sq for sq in near if sq not in (weak, strong)]
    if len(free) < n_pieces:
        return None
    return weak, strong, rng.sample(free, n_pieces)


def generate_position(material: str, rng: random.Random, level: int = MAX_LEVEL, max_tries: int = 5000) -> chess.Board:
    """Random legal position: strong side (random colour) with king + `material`
    against a lone king, strong side to move, no mate in one, strong king not in check,
    weak king placed by `level` (0 corner ... 3 anywhere)."""
    for _ in range(max_tries):
        strong_colour = rng.choice((chess.WHITE, chess.BLACK))
        picked = _level_squares(level, rng, len(material))
        if picked is None:
            continue
        weak, strong, piece_squares = picked
        board = chess.Board(None)
        board.set_piece_at(strong, chess.Piece(chess.KING, strong_colour))
        board.set_piece_at(weak, chess.Piece(chess.KING, not strong_colour))
        ok = True
        for sq, sym in zip(piece_squares, material):
            piece = _PIECE[sym]
            if piece == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            board.set_piece_at(sq, chess.Piece(piece, strong_colour))
        if not ok:
            continue
        board.turn = strong_colour
        board.castling_rights = 0
        if not board.is_valid() or board.is_check() or board.is_game_over():
            continue
        if board.is_attacked_by(strong_colour, weak):
            continue
        if _has_mate_in_one(board):
            continue
        return board
    raise RuntimeError(f"could not generate a position for material {material!r} at level {level}")


class TechniqueSampler:
    """Feeds the finishing slot with technique starts (`FinishStart` with a `config`
    label and the level in `depth`). With `curriculum=False` every start is level 3 and
    `report_label` does nothing."""

    def __init__(self, sets: Sequence[str], caps: Dict[str, int] | None = None, seed: int = 0,
                 curriculum: bool = False, level_start: int = 0, advance_rate: float = 0.7, window: int = 40):
        if not sets:
            raise ValueError("technique sets must not be empty")
        caps = {**DEFAULT_CAPS, **(caps or {})}
        for s in sets:
            if s not in caps:
                raise ValueError(f"no cap for technique set {s!r}")
            if not s or any(ch not in _PIECE for ch in s):
                raise ValueError(f"technique set {s!r} must be pieces from QRBNP")
        if not 0 <= level_start <= MAX_LEVEL:
            raise ValueError(f"technique_level_start must be in [0, {MAX_LEVEL}]")
        if not 0 < advance_rate <= 1 or window < 1:
            raise ValueError("technique_advance_rate must be in (0, 1] and technique_window >= 1")
        self.sets = list(sets)
        self.caps = {s: int(caps[s]) for s in sets}
        self.curriculum = bool(curriculum)
        self.advance_rate = float(advance_rate)
        self.window = int(window)
        self._level = {s: (int(level_start) if curriculum else MAX_LEVEL) for s in sets}
        self._recent = {s: deque(maxlen=self.window) for s in sets}
        self._rng = random.Random(seed)
        self._seed = seed
        self._init = dict(curriculum=curriculum, level_start=level_start, advance_rate=advance_rate, window=window)

    @property
    def current_depth(self) -> int:
        return 0

    def current_level(self, label: str) -> int:
        return self._level[label]

    def levels(self) -> Dict[str, int]:
        return dict(self._level)

    def sample(self) -> FinishStart:
        label = self._rng.choice(self.sets)
        level = self._rng.randint(0, self._level[label]) if self.curriculum else MAX_LEVEL
        board = generate_position(label, self._rng, level)
        winner = WHITE if board.turn == chess.WHITE else BLACK
        return FinishStart(board.fen(), level, self.caps[label], winner, (), label)

    def report(self, depth: int, success: bool) -> None:
        return None

    def report_label(self, label: str, success: bool, clean: bool = True) -> None:
        """Curriculum step: after `window` clean results for `label` (episodes in which
        the demonstrator never acted), advance its level when the success rate reaches
        `advance_rate`; the window then restarts. Episodes with a demonstrator move are
        ignored here (arm CURSIL 2026-09-18: counting them let levels race ahead of skill)."""
        if not self.curriculum or label not in self._recent or not clean:
            return
        recent = self._recent[label]
        recent.append(bool(success))
        if len(recent) == self.window and sum(recent) / self.window >= self.advance_rate:
            if self._level[label] < MAX_LEVEL:
                self._level[label] += 1
            recent.clear()

    def with_seed(self, seed: int) -> "TechniqueSampler":
        return TechniqueSampler(self.sets, self.caps, seed, **self._init)
