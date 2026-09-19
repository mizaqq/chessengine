"""Does the opponent punish our gifts?  (owner's hypothesis, 2026-09-19)

For every ply of the *mover* in sampled games, two rules questions (python-chess,
one ply of look-ahead, no search):
  * mate gift:     after the move the opponent has a mate in one, and some other
                   legal move would not have allowed one;
  * material gift: after the move the opponent can win a piece worth >= 3 net of
                   the immediate recapture, and could not before the move.
A gift counts as *punished* when the opponent's actual reply cashes it in (mates,
or wins >= 3 net material). Reports gift rate per 100 mover plies and the punish
rate per defender. Compare self-play (mover == defender) against the SL prior.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import chess
import torch

from src.model.checkpoints import load_models
from src.viz.play import play_game
from scripts.prepare_puzzles import mating_moves

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def capture_gain(board: chess.Board, move: chess.Move) -> int:
    """Net material the side to move wins with `move` after the cheapest recapture."""
    if not board.is_capture(move):
        return 0
    victim = board.piece_at(move.to_square)
    gained = VALUE[victim.piece_type] if victim else 1        # en passant
    board.push(move)
    recaptures = [m for m in board.legal_moves if m.to_square == move.to_square]
    board.pop()
    if not recaptures:
        return gained
    return gained - VALUE[board.piece_at(move.from_square).piece_type]


def best_gain(board: chess.Board) -> int:
    """Largest net capture gain available to the side to move (0 if none)."""
    return max((capture_gain(board, m) for m in board.legal_moves), default=0)


def gain_if_opponent_moved(board: chess.Board) -> int:
    """Best gain the *other* side would have if it were to move now; 0 when the
    side to move is in check (a null move is then illegal)."""
    if board.is_check():
        return 0
    board.push(chess.Move.null())
    g = best_gain(board)
    board.pop()
    return g


def classify(fen_before: str, san: str, threshold: int = 3) -> dict:
    """Gift flags for one move: {"mate": bool, "material": bool}."""
    before = chess.Board(fen_before)
    move = before.parse_san(san)
    after = before.copy()
    after.push(move)
    out = {"mate": False, "material": False}
    if after.is_game_over():
        return out
    if mating_moves(after):
        for alt in before.legal_moves:
            if alt == move:
                continue
            before.push(alt)
            safe = not before.is_game_over() and not mating_moves(before)
            before.pop()
            if safe:
                out["mate"] = True
                break
    if best_gain(after) >= threshold and best_gain(after) > gain_if_opponent_moved(before):
        out["material"] = True
    return out


def punished(fen_before: str, san: str, reply_san: str | None, gift: dict, threshold: int = 3) -> dict:
    """Did the opponent's actual reply cash the gift in?"""
    if reply_san is None:
        return {k: False for k, v in gift.items() if v}
    board = chess.Board(fen_before)
    board.push_san(san)
    reply = board.parse_san(reply_san)
    out = {}
    if gift["mate"]:
        b = board.copy(); b.push(reply); out["mate"] = b.is_checkmate()
    if gift["material"]:
        out["material"] = capture_gain(board, reply) >= threshold
    return out


def audit_game(game, mover_side: str) -> Counter:
    c = Counter()
    moves = game.moves
    for i, mv in enumerate(moves):
        if mv.side != mover_side:
            continue
        c["plies"] += 1
        gift = classify(mv.fen_before, mv.san)
        reply = moves[i + 1].san if i + 1 < len(moves) else None
        pun = punished(mv.fen_before, mv.san, reply, gift)
        for kind in ("mate", "material"):
            if gift[kind]:
                c[f"gift_{kind}"] += 1
                c[f"punished_{kind}"] += int(pun[kind])
    return c


def audit_pair(mover, defender, games: int, oriented: bool, greedy_defender: bool, seed: int) -> dict:
    torch.manual_seed(seed)
    c = Counter()
    for _ in range(games):
        g = play_game(mover, defender, oriented=oriented, greedy_black=greedy_defender)
        c += audit_game(g, "white")
        g = play_game(defender, mover, oriented=oriented, greedy_white=greedy_defender)
        c += audit_game(g, "black")
    out = {"mover_plies": c["plies"]}
    for kind in ("mate", "material"):
        n, p = c[f"gift_{kind}"], c[f"punished_{kind}"]
        out[f"gift_{kind}_per100"] = 100 * n / c["plies"] if c["plies"] else None
        out[f"gift_{kind}"] = n
        out[f"punish_rate_{kind}"] = p / n if n else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mover", type=Path)
    ap.add_argument("defenders", nargs="+", help="NAME=ckpt_dir; append :greedy for an argmax defender")
    ap.add_argument("--games", type=int, default=20, help="games per colour")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    mover, _, oriented, _ = load_models(args.mover)
    report = {"mover": str(args.mover), "games_per_colour": args.games, "seed": args.seed, "defenders": {}}
    for spec in args.defenders:
        name, path = spec.split("=", 1)
        greedy = path.endswith(":greedy")
        path = path.removesuffix(":greedy")
        defender, _, _, _ = load_models(path)
        r = audit_pair(mover, defender, args.games, oriented, greedy, args.seed)
        report["defenders"][name] = r
        print(f"{name:14s} plies {r['mover_plies']:5d}  mate gifts/100 {r['gift_mate_per100']:.2f} "
              f"punished {r['punish_rate_mate']}  material gifts/100 {r['gift_material_per100']:.2f} "
              f"punished {r['punish_rate_material']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
