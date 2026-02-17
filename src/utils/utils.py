import numpy as np
import chess


def board_from_shaped_observation(obs_list, current_player):
    board = chess.Board(None)

    # Actual plane layout: for each piece type, White then Black (interleaved)
    # kPieceTypes order: King, Queen, Rook, Bishop, Knight, Pawn
    piece_types = [
        chess.KING,
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
        chess.PAWN,
    ]

    for i, piece_type in enumerate(piece_types):
        white_plane = obs_list[2 * i]  # Even planes = White
        black_plane = obs_list[2 * i + 1]  # Odd planes = Black

        for y in range(8):
            for x in range(8):
                # Both OpenSpiel and python-chess: 0 = rank 1 (bottom)
                square = chess.square(x, y)
                if white_plane[y][x] > 0.5:
                    board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
                if black_plane[y][x] > 0.5:
                    board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))

    board.turn = chess.WHITE if current_player == 0 else chess.BLACK
    return board


def piece_difference_from_tensor(states):
    """
    states: shape [N, 20, 8, 8] (batch) or [20, 8, 8] (single).
    Returns tensor of shape [N] with white_score - black_score per board.
    """
    import torch

    if states.dim() == 3:
        states = states.unsqueeze(0)  # [20, 8, 8] → [1, 20, 8, 8]

    values = torch.tensor([0, 9, 5, 3, 3, 1], dtype=states.dtype)

    # Sum each 8x8 plane → [N, 12]
    piece_counts = states[:, :12].sum(dim=(2, 3))

    # Even planes = white, odd planes = black → each [N, 6]
    white_scores = (piece_counts[:, 0::2] * values).sum(dim=1)
    black_scores = (piece_counts[:, 1::2] * values).sum(dim=1)

    return white_scores - black_scores
