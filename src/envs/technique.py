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
from typing import Dict, List, Optional, Sequence

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



# ---- Good-starts curriculum (Florensa et al. 2017, Algorithm 1 / §4.1) -----------------
# Buckets over board geometry: weak king's distance to the nearest edge (0-3), king-to-king
# distance class (2 / 3 / 4+), nearest strong piece's distance to the weak king (1-2 / 3-4 /
# 5+). 36 buckets per material set. A bucket is a "good start" while its clean success
# rate lies in [r_min, r_max]; graduated buckets keep a replay share (the paper's
# starts_old), too-hard ones a probe share (our stand-in for Brownian expansion).
EDGE_CLASSES, KING_CLASSES, PIECE_CLASSES = 4, 3, 3
N_BUCKETS = EDGE_CLASSES * KING_CLASSES * PIECE_CLASSES


def edge_distance(sq: int) -> int:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return min(f, 7 - f, r, 7 - r)


def _king_class(d: int) -> int:
    return 0 if d <= 2 else (1 if d == 3 else 2)


def _piece_class(d: int) -> int:
    return 0 if d <= 2 else (1 if d <= 4 else 2)


def bucket_of(board: chess.Board, strong: bool) -> int:
    weak = board.king(not strong)
    kd = chess.square_distance(weak, board.king(strong))
    pieces = [sq for sq, p in board.piece_map().items() if p.color == strong and p.piece_type != chess.KING]
    pd = min(chess.square_distance(weak, sq) for sq in pieces)
    return min(edge_distance(weak), 3) * 9 + _king_class(kd) * 3 + _piece_class(pd)


def bucket_features(bucket: int):
    """(edge class, king class, piece class) of a bucket id."""
    return bucket // 9, (bucket % 9) // 3, bucket % 3


def generate_position_in_bucket(material: str, bucket: int, rng: random.Random, max_tries: int = 200):
    """Position whose geometry falls in `bucket`, with the usual rejections; None if
    none was found in `max_tries` (an infeasible bucket, e.g. two rooks both within
    two squares of a cornered king without a mate in one)."""
    ec, kc, pc = bucket_features(bucket)
    weak_squares = [sq for sq in range(64) if min(edge_distance(sq), 3) == ec]
    for _ in range(max_tries):
        strong_colour = rng.choice((chess.WHITE, chess.BLACK))
        weak = rng.choice(weak_squares)
        kings = [sq for sq in range(64) if _king_class(chess.square_distance(sq, weak)) == kc
                 and chess.square_distance(sq, weak) >= 2]
        if not kings:
            return None
        strong = rng.choice(kings)
        lo = (1, 3, 5)[pc]
        in_class = [sq for sq in range(64) if sq not in (weak, strong)
                    and _piece_class(chess.square_distance(sq, weak)) == pc]
        others = [sq for sq in range(64) if sq not in (weak, strong)
                  and chess.square_distance(sq, weak) >= lo]
        if not in_class or len(others) < len(material):
            return None
        first = rng.choice(in_class)
        rest_pool = [sq for sq in others if sq != first]
        squares = [first] + rng.sample(rest_pool, len(material) - 1)
        board = chess.Board(None)
        board.set_piece_at(strong, chess.Piece(chess.KING, strong_colour))
        board.set_piece_at(weak, chess.Piece(chess.KING, not strong_colour))
        ok = True
        for sq, sym in zip(squares, material):
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
        if board.is_attacked_by(strong_colour, weak) or _has_mate_in_one(board):
            continue
        return board
    return None


class GoodStartsCurriculum:
    """Per (set, bucket) windowed clean success; draw weights by band membership."""

    # Weight of an unknown bucket by the weak king's edge-distance class: start near the
    # goal (cornered / edge kings) and expand outward, as the paper starts from the goal
    # state itself. Arm GSL (2026-09-18) with a flat prior cold-started at random placement.
    UNKNOWN_PRIOR = (1.0, 0.5, 0.25, 0.1)

    def __init__(self, sets: Sequence[str], r_min: float = 0.1, r_max: float = 0.9, window: int = 20,
                 replay_share: float = 0.2, probe_share: float = 0.1, warm_start: bool = True):
        if not 0 <= r_min < r_max <= 1:
            raise ValueError("need 0 <= technique_r_min < technique_r_max <= 1")
        if window < 1 or replay_share < 0 or probe_share < 0:
            raise ValueError("technique_bucket_window >= 1 and shares >= 0")
        self.r_min, self.r_max, self.window = float(r_min), float(r_max), int(window)
        self.replay_share, self.probe_share = float(replay_share), float(probe_share)
        self.warm_start = bool(warm_start)
        self._recent = {s: [deque(maxlen=self.window) for _ in range(N_BUCKETS)] for s in sets}
        self._infeasible = {s: set() for s in sets}

    def rate(self, label: str, bucket: int):
        r = self._recent[label][bucket]
        return sum(r) / len(r) if len(r) >= self.window else None

    def status(self, label: str, bucket: int) -> str:
        rt = self.rate(label, bucket)
        if rt is None:
            return "unknown"
        if rt > self.r_max:
            return "graduated"
        if rt < self.r_min:
            return "hard"
        return "good"

    def weights(self, label: str) -> List[float]:
        out = []
        for b in range(N_BUCKETS):
            if b in self._infeasible[label]:
                out.append(0.0)
                continue
            st = self.status(label, b)
            unknown_w = self.UNKNOWN_PRIOR[bucket_features(b)[0]] if self.warm_start else 1.0
            out.append({"unknown": unknown_w, "good": 1.0, "graduated": self.replay_share, "hard": self.probe_share}[st])
        return out

    def draw(self, label: str, rng: random.Random) -> int:
        w = self.weights(label)
        if sum(w) <= 0:
            return rng.randrange(N_BUCKETS)
        return rng.choices(range(N_BUCKETS), weights=w, k=1)[0]

    def report(self, label: str, bucket: int, success: bool) -> None:
        self._recent[label][bucket].append(bool(success))

    def mark_infeasible(self, label: str, bucket: int) -> None:
        self._infeasible[label].add(int(bucket))

    def dials(self, label: str) -> Dict[str, int]:
        st = [self.status(label, b) for b in range(N_BUCKETS) if b not in self._infeasible[label]]
        return {"known_buckets": sum(x != "unknown" for x in st), "good_buckets": st.count("good"),
                "graduated": st.count("graduated")}

    def state(self) -> dict:
        return {"recent": {s: [list(d) for d in ds] for s, ds in self._recent.items()},
                "infeasible": {s: sorted(v) for s, v in self._infeasible.items()}}

    def load_state(self, state: dict) -> None:
        for s, lists in state.get("recent", {}).items():
            if s in self._recent:
                for b, vals in enumerate(lists[:N_BUCKETS]):
                    self._recent[s][b] = deque((bool(v) for v in vals), maxlen=self.window)
        for s, vals in state.get("infeasible", {}).items():
            if s in self._infeasible:
                self._infeasible[s] = set(int(v) for v in vals)


class TechniqueSampler:
    """Feeds the finishing slot with technique starts (`FinishStart` with a `config`
    label and the level in `depth`). With `curriculum=False` every start is level 3 and
    `report_label` does nothing."""

    def __init__(self, sets: Sequence[str], caps: Dict[str, int] | None = None, seed: int = 0,
                 curriculum: bool = False, level_start: int = 0, advance_rate: float = 0.7, window: int = 40,
                 mode: Optional[str] = None, r_min: float = 0.1, r_max: float = 0.9, bucket_window: int = 20,
                 replay_share: float = 0.2, probe_share: float = 0.1, warm_start: bool = True):
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
        if mode is None:
            mode = "levels" if curriculum else "off"
        if mode not in ("off", "levels", "good_starts"):
            raise ValueError(f"technique_curriculum must be off, levels or good_starts, got {mode!r}")
        self.mode = mode
        self.sets = list(sets)
        self.caps = {s: int(caps[s]) for s in sets}
        self.curriculum = mode == "levels"
        self.advance_rate = float(advance_rate)
        self.window = int(window)
        self._level = {s: (int(level_start) if self.curriculum else MAX_LEVEL) for s in sets}
        self._recent = {s: deque(maxlen=self.window) for s in sets}
        self.good = GoodStartsCurriculum(sets, r_min, r_max, bucket_window, replay_share, probe_share, warm_start) \
            if mode == "good_starts" else None
        self._rng = random.Random(seed)
        self._seed = seed
        self._init = dict(curriculum=curriculum, level_start=level_start, advance_rate=advance_rate, window=window,
                          mode=mode, r_min=r_min, r_max=r_max, bucket_window=bucket_window,
                          replay_share=replay_share, probe_share=probe_share, warm_start=warm_start)

    @property
    def current_depth(self) -> int:
        return 0

    def current_level(self, label: str) -> int:
        return self._level[label]

    def levels(self) -> Dict[str, int]:
        return dict(self._level)

    def sample(self) -> FinishStart:
        label = self._rng.choice(self.sets)
        if self.good is not None:
            # Good-starts mode: `depth` carries the bucket id; infeasible buckets are retired.
            for _ in range(N_BUCKETS):
                bucket = self.good.draw(label, self._rng)
                board = generate_position_in_bucket(label, bucket, self._rng)
                if board is not None:
                    winner = WHITE if board.turn == chess.WHITE else BLACK
                    return FinishStart(board.fen(), bucket, self.caps[label], winner, (), label)
                self.good.mark_infeasible(label, bucket)
            board = generate_position(label, self._rng, MAX_LEVEL)
            winner = WHITE if board.turn == chess.WHITE else BLACK
            return FinishStart(board.fen(), bucket_of(board, board.turn), self.caps[label], winner, (), label)
        level = self._rng.randint(0, self._level[label]) if self.curriculum else MAX_LEVEL
        board = generate_position(label, self._rng, level)
        winner = WHITE if board.turn == chess.WHITE else BLACK
        return FinishStart(board.fen(), level, self.caps[label], winner, (), label)

    def dials(self) -> Dict[str, int]:
        """Summary keys: technique_level_<set> (levels mode) or the good-starts counts."""
        if self.good is not None:
            out = {}
            for s in self.sets:
                for k, v in self.good.dials(s).items():
                    out[f"technique_{k}_{s}"] = v
            return out
        return {f"technique_level_{s}": v for s, v in self._level.items()}

    def state(self) -> Optional[dict]:
        return {"mode": self.mode, "good": self.good.state()} if self.good is not None else \
            {"mode": self.mode, "levels": dict(self._level)}

    def load_state(self, state: dict) -> None:
        if not state or state.get("mode") != self.mode:
            return
        if self.good is not None and "good" in state:
            self.good.load_state(state["good"])
        elif "levels" in state:
            for s, v in state["levels"].items():
                if s in self._level:
                    self._level[s] = int(v)

    def report(self, depth: int, success: bool) -> None:
        return None

    def report_label(self, label: str, success: bool, clean: bool = True, depth: Optional[int] = None) -> None:
        """Curriculum step: after `window` clean results for `label` (episodes in which
        the demonstrator never acted), advance its level when the success rate reaches
        `advance_rate`; the window then restarts. Episodes with a demonstrator move are
        ignored here (arm CURSIL 2026-09-18: counting them let levels race ahead of skill)."""
        if not clean or label not in self._recent:
            return
        if self.good is not None:
            if depth is not None:
                self.good.report(label, int(depth), success)
            return
        if not self.curriculum:
            return
        recent = self._recent[label]
        recent.append(bool(success))
        if len(recent) == self.window and sum(recent) / self.window >= self.advance_rate:
            if self._level[label] < MAX_LEVEL:
                self._level[label] += 1
            recent.clear()

    def with_seed(self, seed: int) -> "TechniqueSampler":
        return TechniqueSampler(self.sets, self.caps, seed, **self._init)
