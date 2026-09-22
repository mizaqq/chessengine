import torch

from src.model.chess_model import ChessPolicyProbs
from src.training.opponent_pool import OpponentPool, PoolBoards, WHITE, BLACK


def tiny():
    return ChessPolicyProbs(num_filters=8)


def test_pool_fills_and_rolls():
    pool = OpponentPool(tiny(), size=4, snapshot_every=50, seed=0)
    learner = tiny()
    for update in range(1, 251):
        pool.maybe_snapshot(learner, update)
    assert [name for name, _ in pool.members] == ["prior", "snap_100", "snap_150", "snap_200", "snap_250"]


def test_before_first_snapshot_only_prior():
    pool = OpponentPool(tiny(), size=4, snapshot_every=50, seed=0)
    pool.maybe_snapshot(tiny(), 30)
    assert [name for name, _ in pool.members] == ["prior"]
    assert all(pool.sample()[0] == "prior" for _ in range(5))


def test_snapshot_is_frozen_copy():
    pool = OpponentPool(tiny(), size=1, snapshot_every=1, seed=0)
    learner = tiny()
    pool.maybe_snapshot(learner, 1)
    snap = pool.members[-1][1]
    with torch.no_grad():
        for p in learner.parameters():
            p.add_(1.0)
    assert not snap.training
    assert all(not p.requires_grad for p in snap.parameters())
    assert not torch.equal(next(snap.parameters()), next(learner.parameters()))


def test_redraw_only_on_done_boards():
    pool = OpponentPool(tiny(), size=4, snapshot_every=1, seed=0)
    for u in range(1, 5):
        pool.maybe_snapshot(tiny(), u)
    boards = PoolBoards([6, 7, 8], pool, seed=0)
    before = {i: boards.assignment(i) for i in (6, 7, 8)}
    boards.redraw([7])
    assert boards.assignment(6) == before[6] and boards.assignment(8) == before[8]
    # a redraw changes name or colour with high probability over a few tries
    changed = any(boards.redraw([7]) or boards.assignment(7) != before[7] for _ in range(20))
    assert changed


def test_learner_mask_and_grouping():
    pool = OpponentPool(tiny(), size=0, snapshot_every=50, seed=0)
    boards = PoolBoards([0, 2], pool, seed=0)
    boards._state[0] = ("prior", pool.members[0][1], WHITE)
    boards._state[2] = ("prior", pool.members[0][1], BLACK)
    players = torch.tensor([WHITE, WHITE, WHITE, BLACK])
    mask = boards.learner_mask(players)
    # board 0: learner is white and white moves -> learner; board 2: learner black, white moves -> opponent
    assert mask.tolist() == [True, True, False, True]
    groups = boards.opponent_groups(players)
    assert list(groups.values()) == [[2]]


# --- prioritised sampling (openspec/changes/prior-anchored-selfplay/specs/opponent-pool/spec.md) ---

def _pool_with_two_snaps(**kw):
    pool = OpponentPool(tiny(), size=4, snapshot_every=1, seed=0, **kw)
    pool.maybe_snapshot(tiny(), 1)      # snap_1 -> "snap_a"
    pool.maybe_snapshot(tiny(), 2)      # snap_2 -> "snap_b"
    return pool


def test_uniform_sampling_is_the_default_and_equal():
    pool = _pool_with_two_snaps()
    assert pool.sampling == "uniform"
    w = pool.weights()
    assert set(w) == {"prior", "snap_1", "snap_2"} and all(abs(v - 1 / 3) < 1e-9 for v in w.values())


def test_pfsp_weights_follow_one_minus_winrate_squared_plus_floor():
    pool = _pool_with_two_snaps(sampling="pfsp", power=2, min_weight=0.05)
    pool.win_rate.update({"prior": 0.3, "snap_1": 0.5, "snap_2": 1.0})
    w = pool.weights()
    assert abs(w["prior"] - 0.54 / 0.89) < 1e-9        # (0.7^2 + 0.05) / (0.54 + 0.30 + 0.05)
    assert abs(w["snap_1"] - 0.30 / 0.89) < 1e-9
    assert abs(w["snap_2"] - 0.05 / 0.89) < 1e-9
    draws = [pool.sample()[0] for _ in range(4000)]
    assert abs(draws.count("prior") / 4000 - 0.607) < 0.03
    assert draws.count("snap_2") / 4000 < 0.1


def test_record_updates_ema_win_rate_from_half():
    pool = _pool_with_two_snaps(sampling="pfsp")
    assert pool.win_rate["prior"] == 0.5
    pool.record("prior", 0.0)
    assert abs(pool.win_rate["prior"] - 0.45) < 1e-9
    pool.record("prior", 1.0)
    assert abs(pool.win_rate["prior"] - (0.9 * 0.45 + 0.1)) < 1e-9
    pool.record("unknown_member", 1.0)                    # a stale name is ignored, not an error


def test_pfsp_uniform_when_power_zero():
    pool = _pool_with_two_snaps(sampling="pfsp", power=0, min_weight=0.0)
    pool.win_rate.update({"prior": 0.1, "snap_1": 0.9})
    w = pool.weights()
    assert all(abs(v - 1 / 3) < 1e-9 for v in w.values())
