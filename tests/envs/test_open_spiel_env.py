import random

import pytest

from src.envs.open_spiel_env import OpenSpielEnv


def test_game_result_returns_none_when_not_done():
    env = OpenSpielEnv()
    env.reset()
    assert env.game_result() is None


def test_game_result_fools_mate_is_black_win():
    """Fool's mate: 1. f3 e5 2. g4 Qh4# — black wins."""
    import pyspiel
    from src.envs.open_spiel_env import OpenSpielEnv

    game = pyspiel.load_game("chess")
    state = game.new_initial_state()

    env = OpenSpielEnv()
    env.reset()

    moves_san = ["f3", "e5", "g4", "Qh4"]
    for move_san in moves_san:
        legal = state.legal_actions()
        action_map = {state.action_to_string(state.current_player(), a): a for a in legal}
        action = action_map.get(move_san)
        if action is None:
            for k, v in action_map.items():
                if move_san in k:
                    action = v
                    break
        env.step([action])
        state.apply_action(action)

    assert env.is_done()
    assert env.game_result() == "black_win", (
        f"Fool's mate should be black_win, got {env.game_result()}"
    )


def test_game_result_returns_string_when_done():
    env = OpenSpielEnv()
    env.reset()
    for _ in range(1000):
        if env.is_done():
            break
        legal = env.get_legal_actions().nonzero().squeeze(-1).tolist()
        env.step([random.choice(legal)])
    else:
        pytest.fail("Game did not end within 1000 moves")
    result = env.game_result()
    assert result in ("white_win", "black_win", "draw")
