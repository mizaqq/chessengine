"""Harvest mate-in-one positions from the models' own games.

Plays sampled self-play with the newest checkpoint in <ckpt_dir> on a vectorized
env; every position where the side to move has a mating move (rules engine) is
kept, deduplicated by FEN, shuffled with a seed and split into train / held-out
files in the puzzle CSV format.

Usage: python -m scripts.harvest_positions <ckpt_dir> --positions 3300 --holdout 300
       [--boards 32] [--seed 0] [--out-prefix data/puzzles/selfplay_mate_in_1]
"""
import argparse
import csv
import random
from pathlib import Path

import chess
import torch

from scripts.prepare_puzzles import FIELDS, mating_moves
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.model.checkpoints import load_models
from src.model.orientation import orient

WHITE = 1
BLACK = 0


def collect(fens, seen):
    """Return new (fen, mating_moves) rows for positions with a mate available."""
    rows = []
    for fen in fens:
        if fen in seen:
            continue
        mates = mating_moves(chess.Board(fen))
        if mates:
            seen.add(fen)
            rows.append({"puzzle_id": f"sp{len(seen):06d}", "fen": fen,
                         "mating_moves": " ".join(mates), "rating": "0"})
    return rows


def split(rows, holdout, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[holdout:], rows[:holdout]


@torch.no_grad()
def harvest(white, black, oriented, positions, boards, seed):
    torch.manual_seed(seed)
    env = OpenSpielVectorEnv(boards)
    step = env.reset()
    models = {WHITE: white.eval(), BLACK: black.eval()}
    seen, rows = set(), []
    while len(rows) < positions:
        fens = [e.env.get_state.to_string() for e in env.envs]
        rows.extend(collect(fens, seen))
        players = step.current_player
        obs = orient(step.obs, players) if oriented else step.obs
        actions = torch.zeros(boards, dtype=torch.long)
        for pid, model in models.items():
            mask = players == pid
            if mask.any():
                probs, _ = model(obs[mask], step.legal_actions_mask[mask])
                actions[mask] = torch.multinomial(probs, 1).squeeze(1)
        step = env.step(actions)
    return rows


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--positions", type=int, default=3300)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--boards", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="data/puzzles/selfplay_mate_in_1")
    args = ap.parse_args()
    white, black, oriented, paths = load_models(args.ckpt_dir)
    print("harvesting from", paths, "oriented", oriented)
    rows = harvest(white, black, oriented, args.positions, args.boards, args.seed)
    train, held = split(rows, args.holdout, args.seed)
    write_csv(f"{args.out_prefix}_train.csv", train)
    write_csv(f"{args.out_prefix}_eval.csv", held)
    print(f"wrote {len(train)} train and {len(held)} held-out self-play positions")


if __name__ == "__main__":
    main()
