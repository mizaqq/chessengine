import chess

from scripts.harvest_positions import collect, split

ROOK_MATE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_collect_keeps_only_positions_with_a_mate_and_dedups():
    seen = set()
    rows = collect([START, ROOK_MATE, ROOK_MATE], seen)
    assert len(rows) == 1
    assert rows[0]["fen"] == ROOK_MATE and rows[0]["mating_moves"] == "a1a8"
    assert collect([ROOK_MATE], seen) == []


def test_every_collected_row_mates():
    rows = collect([ROOK_MATE, "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"], set())
    for r in rows:
        b = chess.Board(r["fen"])
        for uci in r["mating_moves"].split():
            b.push_uci(uci); assert b.is_checkmate(); b.pop()


def test_split_is_disjoint_and_seeded():
    rows = [{"fen": str(i)} for i in range(20)]
    tr, ho = split(rows, 5, seed=1)
    assert len(tr) == 15 and len(ho) == 5
    assert not {r["fen"] for r in tr} & {r["fen"] for r in ho}
    assert split(rows, 5, seed=1) == (tr, ho)
