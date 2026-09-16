import pytest

from src.envs.start_positions import Puzzle, StartPositionSampler, load_puzzles

FIXTURE = """puzzle_id,fen,mating_moves,rating
p1,7k/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,a1a8,600
p2,6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,a1a8,650
p3,r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1,a8a1,700
"""

PUZZLES = [
    Puzzle("p1", "fen1", ["a1a8"], 600),
    Puzzle("p2", "fen2", ["a1a8"], 650),
    Puzzle("p3", "fen3", ["a8a1"], 700),
]


def test_load_puzzles(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(FIXTURE)
    puzzles = load_puzzles(path)
    assert len(puzzles) == 3
    assert puzzles[0].puzzle_id == "p1"
    assert puzzles[0].fen.startswith("7k/5ppp/")
    assert puzzles[2].mating_moves == ["a8a1"]
    assert puzzles[1].rating == 650


def test_sampler_always_returns_a_puzzle():
    s = StartPositionSampler(PUZZLES, seed=1)
    assert all(s.sample() in {"fen1", "fen2", "fen3"} for _ in range(50))


def test_same_seed_same_sequence():
    a = StartPositionSampler(PUZZLES, seed=7)
    b = StartPositionSampler(PUZZLES, seed=7)
    assert [a.sample() for _ in range(30)] == [b.sample() for _ in range(30)]


def test_with_seed_changes_sequence():
    a = StartPositionSampler(PUZZLES, seed=7)
    b = a.with_seed(8)
    assert [a.sample() for _ in range(30)] != [b.sample() for _ in range(30)]


def test_empty_puzzles_rejected():
    with pytest.raises(ValueError):
        StartPositionSampler([], seed=0)
