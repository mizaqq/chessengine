"""Discounted returns with potential-based material shaping.

Each own step of a model receives the shaped reward
    F = scale * (gamma * Phi(next own position) - Phi(this position))
where Phi is material from that model's perspective and the terminal position has
Phi = 0 (Ng, Harada & Russell 1999). Terminal outcomes arrive via terminal_r_*.
"""
import torch
from torch.testing import assert_close
from src.core.types import StepRecord
from src.model.returns import compute_returns_for_model

WHITE = 1  # OpenSpiel convention
BLACK = 0


def _make_step(player, potential_white, done=False, terminal_r_w=0.0, terminal_r_b=0.0, n=1):
    """potential_white: white-view material of the position the mover saw (pre-move)."""
    return StepRecord(
        player=torch.tensor([player] * n, dtype=torch.long),
        obs=torch.zeros(n, 20, 8, 8),
        legal_mask=torch.ones(n, 4674),
        action=torch.zeros(n, dtype=torch.long),
        old_log_prob=torch.zeros(n),
        old_value=torch.zeros(n),
        entropy=torch.zeros(n),
        num_legal=torch.ones(n) * 20,
        potential_white=torch.tensor([potential_white] * n, dtype=torch.float32),
        done=torch.tensor([done] * n),
        terminal_r_white=torch.tensor([terminal_r_w] * n),
        terminal_r_black=torch.tensor([terminal_r_b] * n),
    )


def _returns(steps, model_id, bootstrap, final_material, gamma=1.0, scale=1.0):
    return compute_returns_for_model(
        steps,
        model_id=model_id,
        bootstrap_value=torch.tensor([bootstrap]),
        gamma=gamma,
        final_material=torch.tensor([final_material]),
        shaping_scale=scale,
    )


# --- shaped reward per own move -------------------------------------------------

def test_single_player_potentials_telescope_to_final_minus_first():
    """All white steps, no done. Shaping telescopes: sum F = gamma^n Phi_T - Phi_0 (gamma=1)."""
    steps = [_make_step(WHITE, 0.0), _make_step(WHITE, 1.0), _make_step(WHITE, 4.0)]
    r = _returns(steps, WHITE, bootstrap=10.0, final_material=6.0)
    # step 2: F = 6 - 4 = 2, R = 2 + 10 = 12
    # step 1: F = 4 - 1 = 3, R = 3 + 12 = 15
    # step 0: F = 1 - 0 = 1, R = 1 + 15 = 16   (= (6 - 0) + 10)
    assert_close(r[2], torch.tensor([12.0]))
    assert_close(r[1], torch.tensor([15.0]))
    assert_close(r[0], torch.tensor([16.0]))


def test_single_player_discounted_matches_hand_computation():
    steps = [_make_step(WHITE, 0.0), _make_step(WHITE, 0.0), _make_step(WHITE, 0.0)]
    r = _returns(steps, WHITE, bootstrap=10.0, final_material=0.0, gamma=0.99)
    # all potentials 0 -> F = 0 everywhere; pure bootstrap discounting
    assert_close(r[2], torch.tensor([9.9]))
    assert_close(r[1], torch.tensor([9.801]))
    assert_close(r[0], torch.tensor([9.70299]))


def test_capture_and_recapture_nets_zero_for_first_mover():
    """White captures (0 -> 3), black recaptures (3 -> 0), white to move again."""
    steps = [
        _make_step(WHITE, 0.0),   # white sees 0, captures
        _make_step(BLACK, 3.0),   # black sees +3 (white view), recaptures
    ]
    rw = _returns(steps, WHITE, bootstrap=0.0, final_material=0.0)
    rb = _returns(steps, BLACK, bootstrap=0.0, final_material=0.0)
    assert_close(rw[0], torch.tensor([0.0]))    # 0 - 0
    assert_close(rb[1], torch.tensor([3.0]))    # black view: 0 - (-3)
    assert_close(rw[1], torch.tensor([0.0]))    # opponent step carries the later R (bootstrap)


def test_discounted_potential():
    """White at material 2, next white decision at material 5, gamma 0.9 -> 0.9*5 - 2 = 2.5."""
    steps = [_make_step(WHITE, 2.0), _make_step(BLACK, 2.0)]
    r = _returns(steps, WHITE, bootstrap=0.0, final_material=5.0, gamma=0.9)
    assert_close(r[0], torch.tensor([2.5]))


def test_scale_applies_to_shaping_only():
    steps = [_make_step(WHITE, 2.0, done=True, terminal_r_w=2.0)]
    r_full = _returns(steps, WHITE, bootstrap=9.0, final_material=5.0, scale=1.0)
    r_half = _returns(steps, WHITE, bootstrap=9.0, final_material=5.0, scale=0.5)
    # done: terminal potential 0 -> F = -2*scale; terminal reward 2 unchanged
    assert_close(r_full[0], torch.tensor([0.0]))
    assert_close(r_half[0], torch.tensor([1.0]))


def test_black_perspective_negates_potential():
    steps = [_make_step(BLACK, -4.0), _make_step(WHITE, -4.0)]
    r = _returns(steps, BLACK, bootstrap=0.0, final_material=-1.0)
    # black view: Phi(s) = 4, Phi(s') = 1 -> F = 1 - 4 = -3 (black lost material)
    assert_close(r[0], torch.tensor([-3.0]))


# --- terminal handling ---------------------------------------------------------

def test_mating_while_ahead_repays_the_loan():
    """White up a queen mates: F = 0 - 9, plus terminal +2 -> -7; bootstrap ignored."""
    steps = [_make_step(WHITE, 9.0, done=True, terminal_r_w=2.0, terminal_r_b=-2.0)]
    r = _returns(steps, WHITE, bootstrap=5.0, final_material=9.0)
    assert_close(r[0], torch.tensor([-7.0]))


def test_opponent_mates_and_new_game_does_not_leak():
    """White moves at 0; black captures a rook and mates (done); a new game follows."""
    steps = [
        _make_step(WHITE, 0.0),
        _make_step(BLACK, 0.0, done=True, terminal_r_w=-2.0, terminal_r_b=2.0),
        _make_step(WHITE, 0.0),   # new game
        _make_step(BLACK, 7.0),   # new game, white already up material
    ]
    r = _returns(steps, WHITE, bootstrap=5.0, final_material=7.0)
    assert_close(r[0], torch.tensor([-2.0]))          # (0 - 0) + (-2)
    assert_close(r[2], torch.tensor([12.0]))          # (7 - 0) + 5, new game only
    # black's mating move: black view Phi = 0 (white view 0 -> black 0), F = 0, terminal +2
    rb = _returns(steps, BLACK, bootstrap=5.0, final_material=7.0)
    assert_close(rb[1], torch.tensor([2.0]))


def test_shaping_sums_to_zero_over_a_complete_game():
    """Any capture sequence from 0 to terminal: return at move 0 equals terminal reward."""
    steps = [
        _make_step(WHITE, 0.0),
        _make_step(BLACK, 1.0),
        _make_step(WHITE, -2.0),
        _make_step(BLACK, 7.0),
        _make_step(WHITE, 7.0),
        _make_step(BLACK, 4.0, done=True, terminal_r_w=-0.5, terminal_r_b=-0.5),
    ]
    rw = _returns(steps, WHITE, bootstrap=3.0, final_material=0.0)
    rb = _returns(steps, BLACK, bootstrap=3.0, final_material=0.0)
    assert_close(rw[0], torch.tensor([-0.5]))
    assert_close(rb[1], torch.tensor([-0.5 - (-1.0)]))  # black's first move saw Phi_b = -1


def test_two_consecutive_opponent_steps_desync():
    """White, black, black, white (desynchronised envs): only own steps get F."""
    steps = [
        _make_step(WHITE, 0.0),
        _make_step(BLACK, 3.0),
        _make_step(BLACK, 3.0),
        _make_step(WHITE, -1.0),
    ]
    r = _returns(steps, WHITE, bootstrap=0.0, final_material=-1.0)
    # step 3: F = -1 - (-1) = 0, R = 0
    # step 0: F = -1 - 0 = -1, R = -1
    assert_close(r[3], torch.tensor([0.0]))
    assert_close(r[0], torch.tensor([-1.0]))
    assert_close(r[1], r[3])   # opponent steps carry the later own step's R
    assert_close(r[2], r[3])


# --- window truncation -----------------------------------------------------------

def test_window_ends_on_opponents_turn_uses_final_material():
    """White's last move from material 1; window ends black-to-move at -2; bootstrap 4."""
    steps = [_make_step(WHITE, 1.0)]
    r = _returns(steps, WHITE, bootstrap=4.0, final_material=-2.0)
    assert_close(r[0], torch.tensor([1.0]))  # (-2 - 1) + 4


# --- shaping off -----------------------------------------------------------------

def test_shaping_none_leaves_only_terminal_and_bootstrap():
    steps = [
        _make_step(WHITE, 0.0),
        _make_step(BLACK, 5.0),
        _make_step(WHITE, -3.0, done=True, terminal_r_w=2.0),
    ]
    r = _returns(steps, WHITE, bootstrap=9.0, final_material=8.0, gamma=0.5, scale=0.0)
    # existing convention: terminal reward enters as gamma * terminal at the ending step
    assert_close(r[2], torch.tensor([1.0]))       # 0.5 * 2
    assert_close(r[0], torch.tensor([0.5]))       # 0.5 * 1.0
