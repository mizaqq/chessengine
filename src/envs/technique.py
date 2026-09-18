"""Generated technique starts: bare-king endings with a forced win by rule.

Owner 2026-09-18: "the bare king stuff should be generated" (the Lichess puzzle
database has none). A material set is the strong side's pieces beyond the king
("Q", "R", "RR", "QR"). The strong side is to move; positions with a mate in one
available, or with the strong king in check, or stalemate, are rejected. Each start
carries the set's ply cap; the board is a draw at the cap (a failure). Perfect-play
worst cases: KQ v K 10 moves, KR v K 16, KRR 7, KQR 6, so caps of 40 / 60 / 40 / 40
plies are about twice the worst case (Florensa et al. 2017 start-state curriculum
with start states where the goal is reachable by rule).
"""
import random
from typing import Dict, Sequence

import chess

from src.envs.start_positions import BLACK, WHITE, FinishStart

DEFAULT_CAPS: Dict[str, int] = {"Q": 40, "R": 60, "RR": 40, "QR": 40}
_PIECE = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}


def _has_mate_in_one(board: chess.Board) -> bool:
    for move in board.legal_moves:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def generate_position(material: str, rng: random.Random, max_tries: int = 1000) -> chess.Board:
    """Random legal position: strong side (random colour) with king + `material`
    against a lone king, strong side to move, no mate in one, strong king not in check."""
    for _ in range(max_tries):
        strong = rng.choice((chess.WHITE, chess.BLACK))
        squares = rng.sample(range(64), 2 + len(material))
        board = chess.Board(None)
        board.set_piece_at(squares[0], chess.Piece(chess.KING, strong))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, not strong))
        if chess.square_distance(squares[0], squares[1]) <= 1:
            continue
        for sq, sym in zip(squares[2:], material):
            piece = _PIECE[sym]
            if piece == chess.PAWN and chess.square_rank(sq) in (0, 7):
                break
            board.set_piece_at(sq, chess.Piece(piece, strong))
        else:
            board.turn = strong
            board.castling_rights = 0
            if not board.is_valid() or board.is_check() or board.is_game_over():
                continue
            # the weak king must not be attacked with the strong side to move (illegal position)
            weak_king = board.king(not strong)
            if board.is_attacked_by(strong, weak_king):
                continue
            if _has_mate_in_one(board):
                continue
            return board
    raise RuntimeError(f"could not generate a position for material {material!r}")


class TechniqueSampler:
    """Feeds the finishing slot with technique starts (`FinishStart` with a `config`
    label). `report()` is a no-op and `current_depth` is 0: there is no depth curriculum,
    the sets are drawn uniformly."""

    current_depth = 0

    def __init__(self, sets: Sequence[str], caps: Dict[str, int] | None = None, seed: int = 0):
        if not sets:
            raise ValueError("technique sets must not be empty")
        caps = {**DEFAULT_CAPS, **(caps or {})}
        for s in sets:
            if s not in caps:
                raise ValueError(f"no cap for technique set {s!r}")
            if not s or any(ch not in _PIECE for ch in s):
                raise ValueError(f"technique set {s!r} must be pieces from QRBNP")
        self.sets = list(sets)
        self.caps = {s: int(caps[s]) for s in sets}
        self._rng = random.Random(seed)

    def sample(self) -> FinishStart:
        label = self._rng.choice(self.sets)
        board = generate_position(label, self._rng)
        winner = WHITE if board.turn == chess.WHITE else BLACK
        return FinishStart(board.fen(), 0, self.caps[label], winner, (), label)

    def report(self, depth: int, success: bool) -> None:
        return None

    def with_seed(self, seed: int) -> "TechniqueSampler":
        return TechniqueSampler(self.sets, self.caps, seed)
