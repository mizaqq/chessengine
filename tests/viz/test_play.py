import chess
import torch

from src.viz.play import play_game


class UniformPolicy(torch.nn.Module):
    """Fake model: uniform over legal moves, value = constant."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, obs, mask):
        probs = mask / mask.sum(dim=-1, keepdim=True)
        return probs, torch.full((obs.shape[0], 1), self.value)


def test_play_game_records_are_consistent():
    game = play_game(UniformPolicy(0.3), UniformPolicy(-0.7), max_moves=12, seed=1)
    assert 1 <= len(game.moves) <= 12
    assert game.result in ("white_win", "black_win", "draw", "unfinished")

    board = chess.Board()
    for i, rec in enumerate(game.moves):
        assert rec.move_number == i + 1
        assert rec.side == ("white" if i % 2 == 0 else "black")
        board.push_san(rec.san)                      # SAN is legal in sequence
        assert rec.fen_after.split(" ")[0] == board.fen().split(" ")[0]
        assert abs(rec.value - (0.3 if rec.side == "white" else -0.7)) < 1e-6
        assert abs(rec.value_plus_material - (rec.value + rec.material)) < 1e-6
        assert len(rec.top_moves) <= 3 and all(0 <= p <= 1 for _, p in rec.top_moves)


def test_material_is_from_movers_perspective():
    game = play_game(UniformPolicy(0.0), UniformPolicy(0.0), max_moves=40, seed=3)
    for rec in game.moves:
        board = chess.Board(rec.fen_before)
        white_mat = sum(
            {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3, chess.PAWN: 1}.get(p.piece_type, 0)
            * (1 if p.color == chess.WHITE else -1)
            for p in board.piece_map().values()
        )
        expected = white_mat if rec.side == "white" else -white_mat
        assert rec.material == expected


def test_greedy_is_deterministic():
    a = play_game(UniformPolicy(0.0), UniformPolicy(0.0), max_moves=6, greedy=True)
    b = play_game(UniformPolicy(0.0), UniformPolicy(0.0), max_moves=6, greedy=True)
    assert [m.san for m in a.moves] == [m.san for m in b.moves]


def test_per_side_greedy_control():
    """White greedy, black sampled: white's replies to identical positions are fixed,
    black's first move varies with the seed."""
    w, b = UniformPolicy(0.0), UniformPolicy(0.0)
    games = [play_game(w, b, greedy_white=True, greedy_black=False, max_moves=2, seed=s) for s in range(6)]
    assert len({g.moves[0].san for g in games}) == 1          # white's first move is fixed
    assert len({g.moves[1].san for g in games}) > 1           # black's varies
