import chess

from src.envs.open_spiel_env import OpenSpielEnv, capture_gain, best_gain, free_capture_actions_of


def _env(fen):
    env = OpenSpielEnv()
    env.reset(fen)
    return env


def _sans(env, actions):
    state = env.env.get_state
    return {state.action_to_string(state.current_player(), a) for a in actions}


def test_hung_queen_is_a_free_capture():
    env = _env("4k3/8/2n5/8/3Q4/8/8/4K3 b - - 0 1")
    assert _sans(env, free_capture_actions_of(env, 3)) == {"Nxd4"}
    assert _sans(env, env.punish_actions()) == {"Nxd4"}


def test_recapture_cancels_the_gift():
    board = chess.Board("4k3/2r5/2n5/8/4B3/8/8/4K3 w - - 0 1")
    assert capture_gain(board, board.parse_san("Bxc6")) == 0
    assert best_gain(board) == 0
    assert free_capture_actions_of(_env(board.fen()), 3) == set()


def test_threshold_excludes_pawn_grabs():
    env = _env("4k3/8/8/3p4/4B3/8/8/4K3 w - - 0 1")   # Bxd5 wins a pawn, nobody recaptures
    assert free_capture_actions_of(env, 1) and free_capture_actions_of(env, 3) == set()


def test_punish_actions_include_mate():
    env = _env("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")   # Ra8# ; no captures
    assert _sans(env, env.punish_actions()) == {"Ra8#"}


def test_mate_takes_priority_over_a_free_capture():
    # White: Ra8 is mate; Bxh4 also wins a free rook. Only the mate is offered.
    env = _env("6k1/5ppp/8/8/7r/8/5B2/R5K1 w - - 0 1")
    assert _sans(env, env.punish_actions()) == {"Ra8#"}
    assert "Bxh4" in _sans(env, free_capture_actions_of(env, 3))
