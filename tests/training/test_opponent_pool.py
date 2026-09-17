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
