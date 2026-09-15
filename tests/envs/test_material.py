"""EnvStep.material: white-view material balance of the returned position.

Piece values: queen 9, rook 5, bishop 3, knight 3, pawn 1, king 0.
"""
import pyspiel
import pytest
import torch

from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv


def _san_to_actions(moves_san):
    """Map a SAN move list to OpenSpiel action ids by replaying on a fresh state."""
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    actions = []
    for san in moves_san:
        legal = state.legal_actions()
        by_name = {state.action_to_string(state.current_player(), a): a for a in legal}
        action = by_name.get(san)
        if action is None:
            action = next(v for k, v in by_name.items() if san in k)
        actions.append(action)
        state.apply_action(action)
    return actions


# 1. e4 d5 2. exd5 : white pawn captures black pawn -> material +1
PAWN_CAPTURE = ["e4", "d5", "exd5"]


@pytest.fixture(params=["sync", "async"])
def make_env(request):
    created = []

    def _make(n):
        env = (
            OpenSpielVectorEnv(n) if request.param == "sync" else OpenSpielAsyncVectorEnv(n)
        )
        created.append(env)
        return env

    yield _make
    for env in created:
        if hasattr(env, "close"):
            env.close()


def test_material_is_zero_after_reset(make_env):
    env = make_env(2)
    step = env.reset()
    assert step.material.shape == (2,)
    assert torch.equal(step.material, torch.zeros(2))


def test_material_after_pawn_capture(make_env):
    env = make_env(1)
    step = env.reset()
    for action in _san_to_actions(PAWN_CAPTURE):
        step = env.step(torch.tensor([action]))
    assert step.material.tolist() == [1.0]


def test_env_step_has_no_per_step_reward(make_env):
    env = make_env(1)
    step = env.reset()
    assert not hasattr(step, "reward")
