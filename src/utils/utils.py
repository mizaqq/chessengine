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
