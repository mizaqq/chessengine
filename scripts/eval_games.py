"""Play sampled games from the opening and report mate-in-one rate and results.

Usage: python -m scripts.eval_games <ckpt_dir> [--games 40] [--seed 0] [--out games.json]
Mate-in-one rate: over all positions where the mover had a mating move available,
the share where it played one. Both sides sample from the policy.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import chess
import torch

from src.model.checkpoints import load_models
from scripts.prepare_puzzles import mating_moves
from src.viz.play import play_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    white, black, oriented, paths = load_models(args.ckpt_dir)

    results = Counter()
    opportunities = 0
    found = 0
    lengths = []
    torch.manual_seed(args.seed)
    for _ in range(args.games):
        game = play_game(white, black, greedy=False, oriented=oriented)
        results[game.result] += 1
        lengths.append(len(game.moves))
        for move in game.moves:
            board = chess.Board(move.fen_before)
            mates = mating_moves(board)
            if not mates:
                continue
            opportunities += 1
            played = chess.Board(move.fen_before)
            played.push_san(move.san)
            found += int(played.is_checkmate())

    total = sum(results.values())
    summary = {
        "games": total,
        "results": dict(results),
        "draw_rate": results["draw"] / total,
        "decisive_rate": (results["white_win"] + results["black_win"]) / total,
        "mate_in_one_opportunities": opportunities,
        "mate_in_one_found": found,
        "mate_in_one_rate": found / opportunities if opportunities else None,
        "mean_plies": sum(lengths) / total,
        "checkpoints": paths,
        "oriented": oriented,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
