import io

import chess
import chess.pgn

from scripts.prepare_finishes import collect, finish_record, split, write_csv
from src.envs.start_positions import BLACK, WHITE, load_finishes

SCHOLAR = """[Event "x"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1550"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""
RESIGN = """[Event "x"]
[Result "0-1"]
[WhiteElo "1600"]
[BlackElo "1550"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 0-1
"""
FOOLS = """[Event "x"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "1500"]

1. f3 e5 2. g4 Qh4# 0-1
"""
LOW = SCHOLAR.replace('[WhiteElo "1600"]', '[WhiteElo "1200"]')


def _game(pgn):
    return chess.pgn.read_game(io.StringIO(pgn))


def test_short_mate_game_anchors_at_initial_position():
    rec = finish_record(_game(SCHOLAR), "g1", window=40)
    assert rec.anchor_fen == chess.STARTING_FEN
    assert rec.moves == ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
    assert rec.winner == WHITE
    assert rec.max_depth() == 6


def test_black_mate_has_black_winner():
    rec = finish_record(_game(FOOLS), "g2")
    assert rec.winner == BLACK and rec.moves[-1] == "d8h4"


def test_resignation_rejected():
    assert finish_record(_game(RESIGN), "g3") is None


def test_window_anchors_forty_plies_before_the_end():
    # 7-ply game with window 4: anchor after 3 plies, 4 moves stored.
    rec = finish_record(_game(SCHOLAR), "g1", window=4)
    assert len(rec.moves) == 4 and rec.moves[-1] == "h5f7"
    b = chess.Board(rec.anchor_fen)
    for u in rec.moves:
        b.push_uci(u)
    assert b.is_checkmate()


def test_collect_applies_rating_floor_and_mate_filter():
    recs = collect(io.StringIO("\n".join([SCHOLAR, RESIGN, FOOLS, LOW])), min_elo=1500, total=10, window=40)
    assert [r.winner for r in recs] == [WHITE, BLACK]


def test_csv_round_trip(tmp_path):
    recs = collect(io.StringIO("\n".join([SCHOLAR, FOOLS])), min_elo=1500, total=10, window=40)
    train, ev = split(recs, n_eval=1, seed=0)
    write_csv(tmp_path / "f.csv", train + ev)
    back = load_finishes(tmp_path / "f.csv")
    assert {r.game_id for r in back} == {r.game_id for r in recs}
    assert all(r.moves and r.winner in (WHITE, BLACK) for r in back)
