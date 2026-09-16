import csv
from pathlib import Path

import chess
import pytest

from scripts.prepare_puzzles import convert_row, mating_moves

# Lichess-style row: FEN before the opponent's move, Moves = opponent move + mate.
ROW = {
    "PuzzleId": "abc12",
    "FEN": "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1",
    "Moves": "g8h8 a1a8",
    "Rating": "600",
    "Themes": "mate mateIn1 endgame",
}


def test_convert_row_applies_first_move_and_lists_mates():
    out = convert_row(ROW)
    assert out["puzzle_id"] == "abc12"
    assert out["fen"].startswith("7k/5ppp/8/8/8/8/5PPP/R5K1 w")
    assert out["mating_moves"] == "a1a8"
    assert out["rating"] == "600"


def test_convert_row_rejects_non_mate_in_one_theme():
    assert convert_row({**ROW, "Themes": "fork middlegame"}) is None


def test_mating_moves_from_board():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert mating_moves(board) == ["a1a8"]


DATA = Path("data/puzzles")


@pytest.mark.skipif(not (DATA / "mate_in_1_train.csv").exists(), reason="puzzle files not built")
def test_written_files_are_disjoint_and_sized():
    def ids(name):
        with open(DATA / name) as fh:
            return [r["puzzle_id"] for r in csv.DictReader(fh)]
    train, ev = ids("mate_in_1_train.csv"), ids("mate_in_1_eval.csv")
    assert len(train) == 20000 and len(ev) == 2000
    assert not set(train) & set(ev)


@pytest.mark.skipif(not (DATA / "mate_in_1_eval.csv").exists(), reason="puzzle files not built")
def test_every_eval_row_is_mate_in_one():
    with open(DATA / "mate_in_1_eval.csv") as fh:
        for r in csv.DictReader(fh):
            board = chess.Board(r["fen"])
            for uci in r["mating_moves"].split():
                board.push_uci(uci)
                assert board.is_checkmate(), r["puzzle_id"]
                board.pop()
