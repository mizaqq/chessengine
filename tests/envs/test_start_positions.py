import pytest

from src.envs.start_positions import FenSampler, Puzzle, StartPositionSampler, load_fens, load_puzzles

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
    assert puzzles[2].mating_moves == ["a8a1"] and puzzles[2].key_moves == ["a8a1"]
    assert puzzles[1].rating == 650
    assert all(p.mate_in == 1 for p in puzzles)


def test_load_puzzles_new_header_with_depth(tmp_path):
    path = tmp_path / "p2.csv"
    path.write_text("puzzle_id,fen,key_moves,mate_in,rating\nq1,7k/8/5K2/8/8/8/8/1R6 w - - 0 1,f6g6,2,900\n")
    (pz,) = load_puzzles(path)
    assert pz.mate_in == 2 and pz.key_moves == ["f6g6"]


def test_sampler_always_returns_a_puzzle():
    s = StartPositionSampler(PUZZLES, seed=1)
    assert all(s.sample().fen in {"fen1", "fen2", "fen3"} for _ in range(50))


def test_same_seed_same_sequence():
    a = StartPositionSampler(PUZZLES, seed=7)
    b = StartPositionSampler(PUZZLES, seed=7)
    assert [a.sample().fen for _ in range(30)] == [b.sample().fen for _ in range(30)]


def test_with_seed_changes_sequence():
    a = StartPositionSampler(PUZZLES, seed=7)
    b = a.with_seed(8)
    assert [a.sample().fen for _ in range(30)] != [b.sample().fen for _ in range(30)]


def test_empty_puzzles_rejected():
    with pytest.raises(ValueError):
        StartPositionSampler([], seed=0)


def test_mixed_sampler_equal_weights():
    from src.envs.start_positions import MixedSampler
    a = [Puzzle("a", "fenA", [], 0)]
    b = [Puzzle("b", "fenB", [], 0)]
    s = MixedSampler([(a, 0.5), (b, 0.5)], seed=0)
    draws = [s.sample().fen for _ in range(1000)]
    assert 400 < draws.count("fenA") < 600


def test_mixed_sampler_zero_weight_and_determinism():
    from src.envs.start_positions import MixedSampler
    a = [Puzzle("a", "fenA", [], 0)]
    b = [Puzzle("b", "fenB", [], 0)]
    s = MixedSampler([(a, 1.0), (b, 0.0)], seed=3)
    assert all(s.sample().fen == "fenA" for _ in range(50))
    x, y = MixedSampler([(a, 1), (b, 1)], seed=5), MixedSampler([(a, 1), (b, 1)], seed=5)
    assert [x.sample().fen for _ in range(40)] == [y.sample().fen for _ in range(40)]


def test_mixed_sampler_rejects_bad_weights():
    from src.envs.start_positions import MixedSampler
    a = [Puzzle("a", "fenA", [], 0)]
    with pytest.raises(ValueError):
        MixedSampler([(a, -1.0)])
    with pytest.raises(ValueError):
        MixedSampler([([], 1.0)])


def test_fen_sampler_and_loader(tmp_path):
    path = tmp_path / "mid.csv"; path.write_text("fen\nfenA\nfenB\n")
    fens = load_fens(path)
    assert fens == ["fenA", "fenB"]
    s = FenSampler(fens, seed=3)
    assert all(s.sample() in fens for _ in range(20))
    assert [s.with_seed(1).sample() for _ in range(10)] == [FenSampler(fens, 1).sample() for _ in range(10)]
    with pytest.raises(ValueError):
        FenSampler([])
