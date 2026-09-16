"""Orient OpenSpiel chess observations to the side to move (AlphaZero style).

OpenSpiel's 20 planes are absolute: planes 0-11 are (white, black) pairs per piece
type, 12 empty squares, 13 repetitions, 14 side to move (white = 1), 15
irreversible-move counter, 16-19 castling rights (white left, white right, black
left, black right). Rank index 0 is white's back rank and the board is never
flipped for black. For black to move we reverse the ranks and swap each white
plane with its black twin, so the network always sees "my pieces" first and my
back rank at index 0. Files are not mirrored (chess is not left-right symmetric).
Move indices need no change: OpenSpiel already encodes them relative to the mover.
"""
import torch

WHITE = 1
BLACK = 0

# Plane i of the oriented tensor is plane PLANE_PERMUTATION[i] of the original.
PLANE_PERMUTATION = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,   # piece pairs swapped
                     12, 13, 14, 15,                          # empty, reps, side, counter
                     18, 19, 16, 17]                          # castling: black<->white
_PERM = torch.tensor(PLANE_PERMUTATION)


def orient_black(obs: torch.Tensor) -> torch.Tensor:
    """Orient black-to-move observations (..., 20, 8, 8); its own inverse."""
    return obs.flip(-2)[..., _PERM, :, :]


def orient(obs: torch.Tensor, players: torch.Tensor) -> torch.Tensor:
    """Orient a batch (N, 20, 8, 8) given each row's side to move (1 white, 0 black)."""
    out = obs.clone()
    black = players == BLACK
    if black.any():
        out[black] = orient_black(obs[black])
    return out
