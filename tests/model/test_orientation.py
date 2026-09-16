import chess
import numpy as np
import pyspiel
import torch

from src.envs.start_positions import load_puzzles
from src.model.orientation import BLACK, WHITE, orient, orient_black

GAME = pyspiel.load_game("chess")


def obs_of(fen: str) -> torch.Tensor:
    state = GAME.new_initial_state(fen)
    player = state.current_player()
    return torch.tensor(np.array(state.observation_tensor(player), dtype=np.float32)).reshape(20, 8, 8)


def sq(plane: torch.Tensor, square: str) -> float:
    """Value at a square named in algebraic notation, rank index 0 = rank 1."""
    file = "abcdefgh".index(square[0])
    rank = int(square[1]) - 1
    return float(plane[rank, file])


START_W = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_B = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"


def test_white_observation_unchanged():
    obs = obs_of(START_W).unsqueeze(0)
    assert torch.equal(orient(obs, torch.tensor([WHITE])), obs)


def test_mirrored_start_positions_identical_except_side_plane():
    w = obs_of(START_W)
    b = orient_black(obs_of(START_B))
    same = [p for p in range(20) if torch.equal(w[p], b[p])]
    assert sorted(set(range(20)) - set(same)) == [14]


def test_black_pawn_a7_becomes_my_pawn_a2():
    # Only pawn on the board is black's on a7; kings on e1/e8; black to move.
    obs = orient_black(obs_of("4k3/p7/8/8/8/8/8/4K3 b - - 0 1"))
    my_pawn, their_pawn = obs[10], obs[11]      # after the swap: mover's pawns first
    assert sq(my_pawn, "a2") == 1.0 and my_pawn.sum() == 1.0
    assert their_pawn.sum() == 0.0


def test_castling_rights_follow_the_swap():
    # Only black may castle kingside; black to move.
    obs = orient_black(obs_of("4k2r/8/8/8/8/8/8/4K3 b k - 0 1"))
    assert obs[16].mean() == 0.0   # my left  (queen side)
    assert obs[17].mean() == 1.0   # my right (king side)
    assert obs[18].mean() == 0.0 and obs[19].mean() == 0.0


def test_black_orientation_is_its_own_inverse():
    obs = obs_of(START_B)
    assert torch.equal(orient_black(orient_black(obs)), obs)


def test_batched_orient_only_touches_black_rows():
    w, b = obs_of(START_W), obs_of(START_B)
    out = orient(torch.stack([w, b]), torch.tensor([WHITE, BLACK]))
    assert torch.equal(out[0], w)
    assert torch.equal(out[1], orient_black(b))


def test_legal_masks_of_colour_mirrored_positions_coincide():
    puzzles = load_puzzles("data/puzzles/mate_in_1_eval.csv")[:20]
    for puzzle in puzzles:
        mirrored = chess.Board(puzzle.fen).mirror().fen()   # flips ranks, swaps colours & castling
        a = GAME.new_initial_state(puzzle.fen)
        b = GAME.new_initial_state(mirrored)
        assert a.legal_actions() == b.legal_actions(), puzzle.puzzle_id
