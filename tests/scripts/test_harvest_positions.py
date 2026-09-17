from scripts.harvest_positions import collect, write_csv
from src.envs.start_positions import load_puzzles


def test_harvested_rows_use_the_current_puzzle_columns(tmp_path):
    rows = collect(["6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"], set())
    assert len(rows) == 1 and rows[0]["key_moves"] == "a1a8" and rows[0]["mate_in"] == "1"
    write_csv(tmp_path / "h.csv", rows)
    p = load_puzzles(tmp_path / "h.csv")
    assert p[0].key_moves == ["a1a8"] and p[0].mate_in == 1
