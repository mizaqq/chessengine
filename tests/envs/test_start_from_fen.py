import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv, _material_np
from src.envs.start_positions import Puzzle, StartPositionSampler

ROOK_MATE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"   # Ra8# available, white +5
BLACK_MATE = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"  # Ra1# available, black to move


def _winning_terminal_actions(env: OpenSpielEnv):
    state = env.env.get_state
    player = state.current_player()
    out = []
    for a in state.legal_actions():
        child = state.child(a)
        if child.is_terminal() and child.returns()[player] > 0:
            out.append(a)
    return out


def test_reset_from_rook_mate_position():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE)
    assert env.get_current_player() == 1  # white
    assert _material_np(env.state()) == 5.0
    assert env.get_legal_actions().sum() == 20
    assert len(_winning_terminal_actions(env)) == 1
    assert not env.is_done()


def test_reset_from_fen_black_to_move():
    env = OpenSpielEnv()
    env.reset(BLACK_MATE)
    assert env.get_current_player() == 0
    assert _material_np(env.state()) == -5.0


def test_reset_without_fen_is_opening():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE)
    env.reset()
    assert env.get_current_player() == 1
    assert _material_np(env.state()) == 0.0
    assert env.get_legal_actions().sum() == 20


def test_playing_the_mate_ends_the_game_as_white_win():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE)
    (mate,) = _winning_terminal_actions(env)
    env.step(mate)
    assert env.is_done()
    assert env.game_result() == "white_win"


PUZZLES = [Puzzle("p1", ROOK_MATE, ["a1a8"], 600)]


def test_vector_env_fraction_one_starts_from_puzzles():
    env = OpenSpielVectorEnv(4, StartPositionSampler(PUZZLES, 1.0, seed=0))
    step = env.reset()
    assert (step.material == 5.0).all()
    assert (step.current_player == 1).all()


def test_vector_env_fraction_zero_is_opening():
    env = OpenSpielVectorEnv(2, StartPositionSampler(PUZZLES, 0.0, seed=0))
    step = env.reset()
    assert (step.material == 0.0).all()


def test_vector_env_auto_reset_uses_sampler():
    env = OpenSpielVectorEnv(1, StartPositionSampler(PUZZLES, 1.0, seed=0))
    env.reset()
    (mate,) = _winning_terminal_actions(env.envs[0])
    step = env.step(torch.tensor([mate]))
    assert step.done[0]
    assert step.info["game_results"][0] == "white_win"
    assert step.material[0] == 5.0  # new game started from the puzzle again


def test_async_env_fraction_one_and_auto_reset():
    from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv
    env = OpenSpielAsyncVectorEnv(2, StartPositionSampler(PUZZLES, 1.0, seed=0))
    try:
        step = env.reset()
        assert (step.material == 5.0).all()
        probe = OpenSpielEnv()
        probe.reset(ROOK_MATE)
        (mate,) = _winning_terminal_actions(probe)
        step = env.step(torch.tensor([mate, mate]))
        assert step.done.all()
        assert step.info["game_results"] == {0: "white_win", 1: "white_win"}
        assert (step.material == 5.0).all()
    finally:
        env.close()
