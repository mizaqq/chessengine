import chess

from scripts.blunder_audit import classify, punished, capture_gain


def test_hanging_queen_is_a_material_gift_and_capture_punishes():
    # White queen steps to d4, attacked by the c6 knight and undefended.
    fen = "4k3/8/2n5/8/8/8/8/Q3K3 w - - 0 1"
    gift = classify(fen, "Qd4")
    assert gift["material"] and not gift["mate"]
    assert punished(fen, "Qd4", "Nxd4", gift)["material"]
    assert not punished(fen, "Qd4", "Kf8", gift)["material"]


def test_allowing_mate_in_one_is_a_mate_gift_when_avoidable():
    # Black to move. Kh8 allows Ra8#; Kf8 would have left e7 as an escape square.
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1"
    assert classify(fen, "Kh8")["mate"]
    assert not classify(fen, "Kf8")["mate"]
    assert punished(fen, "Kh8", "Ra8#", classify(fen, "Kh8"))["mate"]


def test_recapture_removes_the_gain():
    board = chess.Board("4k3/2r5/2n5/8/4B3/8/8/4K3 w - - 0 1")
    assert capture_gain(board, board.parse_san("Bxc6")) == 0      # knight 3, bishop lost to Rxc6
    board = chess.Board("4k3/8/2n5/8/4B3/8/8/4K3 w - - 0 1")
    assert capture_gain(board, board.parse_san("Bxc6")) == 3      # nobody recaptures
