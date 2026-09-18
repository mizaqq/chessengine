import json
import random

import chess
import pytest

from src.envs.technique import (GoodStartsCurriculum, TechniqueSampler, N_BUCKETS, bucket_of, bucket_features,
                                generate_position_in_bucket, _has_mate_in_one)


def test_band_selects_learnable_starts_scenario():
    c = GoodStartsCurriculum(["Q"], r_min=0.1, r_max=0.9, window=20, replay_share=0.2, probe_share=0.1,
                             warm_start=False)
    A, B, C, D = 5, 6, 7, 8
    for i in range(20):
        c.report("Q", A, i < 19)      # 0.95
        c.report("Q", B, i < 10)      # 0.5
        c.report("Q", C, i < 0)       # 0.0 (below r_min)
    w = c.weights("Q")
    assert w[A] == 0.2 and w[B] == 1.0 and w[C] == 0.1 and w[D] == 1.0
    assert c.status("Q", A) == "graduated" and c.status("Q", B) == "good" and c.status("Q", C) == "hard"
    assert c.dials("Q") == {"known_buckets": 3, "good_buckets": 1, "graduated": 1}


def test_geometry_honoured_scenario():
    rng = random.Random(0)
    bucket = 0 * 9 + 0 * 3 + 0        # edge distance 0, king distance 2, piece within 2
    for _ in range(10):
        b = generate_position_in_bucket("R", bucket, rng)
        assert b is not None
        strong = b.turn
        weak = b.king(not strong)
        f, r = chess.square_file(weak), chess.square_rank(weak)
        assert min(f, 7 - f, r, 7 - r) == 0
        assert chess.square_distance(weak, b.king(strong)) == 2
        rook = next(sq for sq, p in b.piece_map().items() if p.piece_type == chess.ROOK)
        assert chess.square_distance(weak, rook) <= 2
        assert bucket_of(b, strong) == bucket and not _has_mate_in_one(b) and not b.is_check()


def test_bucket_roundtrip_and_infeasible_bucket_retired():
    for bkt in range(N_BUCKETS):
        e, k, pc = bucket_features(bkt)
        assert bkt == e * 9 + k * 3 + pc
    s = TechniqueSampler(["Q"], seed=0, mode="good_starts")
    starts = [s.sample() for _ in range(40)]
    assert all(0 <= st.depth < N_BUCKETS and st.config == "Q" for st in starts)
    # every sampled start's board really is in its reported bucket
    for st in starts:
        b = chess.Board(st.fen)
        assert bucket_of(b, b.turn) == st.depth


def test_teacher_made_wins_ignored_and_state_roundtrip():
    s = TechniqueSampler(["Q"], seed=0, mode="good_starts", bucket_window=4)
    for _ in range(10):
        s.report_label("Q", True, clean=False, depth=3)
    assert s.good.rate("Q", 3) is None
    for i in range(4):
        s.report_label("Q", i < 2, clean=True, depth=3)
    assert s.good.rate("Q", 3) == 0.5
    state = json.loads(json.dumps(s.state()))
    t = TechniqueSampler(["Q"], seed=1, mode="good_starts", bucket_window=4)
    t.load_state(state)
    assert t.good.rate("Q", 3) == 0.5 and t.dials()["technique_good_buckets_Q"] == 1


def test_bad_band_rejected():
    with pytest.raises(ValueError):
        GoodStartsCurriculum(["Q"], r_min=0.9, r_max=0.1)


def test_warm_start_weights_unknown_buckets_by_edge_distance():
    c = GoodStartsCurriculum(["Q"])
    w = c.weights("Q")
    assert w[0] == 1.0 and w[9] == 0.5 and w[18] == 0.25 and w[27] == 0.1
    for i in range(20):
        c.report("Q", 27, i < 10)          # a known good bucket far from the edge weighs 1
    assert c.weights("Q")[27] == 1.0
