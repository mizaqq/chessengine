"""Concatenate position sets written by scripts/prepare_games.py (same keys, row-aligned).

    python -m scripts.merge_npz --out data/games/human_2m_train.npz data/games/human_big_train.npz data/games/human_2014-01_train.npz
"""
import argparse
import numpy as np


def merge(paths, out):
    parts = [np.load(p) for p in paths]
    keys = parts[0].files
    for p in parts[1:]:
        if p.files != keys:
            raise ValueError(f"key mismatch: {keys} vs {p.files}")
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    np.savez_compressed(out, **arrays)
    return {k: v.shape for k, v in arrays.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(merge(a.inputs, a.out))


if __name__ == "__main__":
    main()
