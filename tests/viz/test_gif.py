from types import SimpleNamespace

import chess
from PIL import Image

from src.viz.gif import game_to_gif


def _game():
    b = chess.Board()
    moves = []
    for i, san in enumerate(["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]):
        before = b.fen(); b.push_san(san)
        moves.append(SimpleNamespace(move_number=i // 2 + 1, side="white" if i % 2 == 0 else "black", san=san,
                                     fen_before=before, fen_after=b.fen(), prob_played=0.5, value=0.1,
                                     material=0.0, value_plus_material=0.1, top_moves=[]))
    return SimpleNamespace(moves=moves, result="white_win")


def test_gif_has_one_frame_per_move_plus_start(tmp_path):
    path = game_to_gif(_game(), tmp_path / "g.gif", size=160, ms=100)
    im = Image.open(path)
    assert im.format == "GIF" and im.n_frames == 8
