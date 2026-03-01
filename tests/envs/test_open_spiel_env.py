from src.envs.open_spiel_env import OpenSpielEnv


def test_game_result_returns_none_when_not_done():
    env = OpenSpielEnv()
    env.reset()
    assert env.game_result() is None


def test_game_result_returns_string_when_done():
    env = OpenSpielEnv()
    env.reset()
    import random
    while not env.is_done():
        legal = env.get_legal_actions().nonzero().squeeze(-1).tolist()
        env.step([random.choice(legal)])
    result = env.game_result()
    assert result in ("white_win", "black_win", "draw")
