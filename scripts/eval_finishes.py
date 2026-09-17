"""Held-out finishing conversion rates for a checkpoint directory.

Usage: python -m scripts.eval_finishes <ckpt_dir> [--records data/finishes/finish_eval.csv]
         [--depths 2 10 20] [--games 100] [--cap-margin 20] [--seed 0] [--out finishes.json]
"""
import argparse
import json
from pathlib import Path

from src.envs.start_positions import load_finishes
from src.eval.finishes import evaluate_finishes
from src.model.checkpoints import load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--records", default="data/finishes/finish_eval.csv")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 10, 20])
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--cap-margin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    white, black, oriented, paths = load_models(args.ckpt_dir)
    result = evaluate_finishes(white, black, load_finishes(args.records), depths=args.depths, games=args.games,
                               cap_margin=args.cap_margin, oriented=oriented, seed=args.seed)
    summary = {**result, "checkpoints": paths, "records": args.records, "seed": args.seed}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
