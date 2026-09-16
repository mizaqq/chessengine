"""Evaluate saved checkpoints on the held-out mate-in-one set.

Usage: python -m scripts.eval_checkpoint_puzzles <ckpt_dir> [--eval data/puzzles/mate_in_1_eval.csv] [--out baseline.json]
Picks the newest white_/black_ checkpoint pair in <ckpt_dir>.
"""
import argparse
import glob
import json
from pathlib import Path

import torch

from src.envs.start_positions import load_puzzles
from src.eval.puzzles import evaluate_mate_in_one
from src.model.checkpoints import load_models


def load_pair(ckpt_dir: Path):
    """(white, black, white_path, black_path); kept for callers, see load_models for `oriented`."""
    w, b, _oriented, paths = load_models(ckpt_dir)
    return w, b, paths.get("white", paths.get("model")), paths.get("black", paths.get("model"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--eval", default="data/puzzles/mate_in_1_eval.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    white, black, oriented, paths = load_models(args.ckpt_dir)
    result = evaluate_mate_in_one(white, black, load_puzzles(args.eval), oriented=oriented)
    result.update({"checkpoints": paths, "oriented": oriented, "eval_file": args.eval})
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
