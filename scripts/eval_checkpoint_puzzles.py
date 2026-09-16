"""Evaluate saved checkpoints on the held-out mate-in-one set.

Usage: python scripts/eval_checkpoint_puzzles.py <ckpt_dir> [--eval data/puzzles/mate_in_1_eval.csv] [--out baseline.json]
Picks the newest white_/black_ checkpoint pair in <ckpt_dir>.
"""
import argparse
import glob
import json
from pathlib import Path

import torch

from src.envs.start_positions import load_puzzles
from src.eval.puzzles import evaluate_mate_in_one
from src.model.chess_model import ChessPolicyProbs


def load_pair(ckpt_dir: Path):
    white = sorted(glob.glob(str(ckpt_dir / "white_model_*.pth")))[-1]
    black = sorted(glob.glob(str(ckpt_dir / "black_model_*.pth")))[-1]
    models = []
    for path in (white, black):
        m = ChessPolicyProbs()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models.append(m)
    return models[0], models[1], white, black


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--eval", default="data/puzzles/mate_in_1_eval.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    white, black, wpath, bpath = load_pair(args.ckpt_dir)
    result = evaluate_mate_in_one(white, black, load_puzzles(args.eval))
    result.update({"white_ckpt": wpath, "black_ckpt": bpath, "eval_file": args.eval})
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
