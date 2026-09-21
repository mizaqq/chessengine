"""Checkpoint-vs-checkpoint win matrix (the "is self-play cycling?" probe).

Bansal et al. 2017 (ICLR 2018, Table 1) judge opponent-sampling strategies by a
win-rate matrix between the trained agents; "the policy at any time should be able
to defeat random older versions of itself, thus ensuring continual learning". If a
newer checkpoint loses to an older one here, plain self-play has cycled and an
opponent pool / NFSP is worth its cost; if newer beats older, it is not urgent.

Each unordered pair plays `--games` games per colour, both sides sampling from
their policy (evaluation convention of scripts/eval_games.py). Score from the row
player's side: win 1, draw or unfinished (300-move cap) 0.5, loss 0.

Usage: python -m scripts.eval_matrix name=dir name=dir ... [--games 30] [--seed 0]
         [--workers 4] [--out matrix.json]
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from src.model.checkpoints import load_models

from src.eval.matches import play_match, score_from_counts  # noqa: E402


def play_pair(job):
    """One pairing, both colours. Returns counts from the first name's side."""
    (a, dir_a), (b, dir_b), games, seed = job[:4]
    start_fens = job[4] if len(job) > 4 else None
    torch.set_num_threads(1)
    ma, _, oa, _ = load_models(dir_a)
    mb, _, ob, _ = load_models(dir_b)
    if not (oa and ob):
        raise ValueError("matrix expects shared (oriented) checkpoints")
    return a, b, play_match(ma, mb, games, seed, start_fens=start_fens)


def build_table(names, pair_results):
    """Full matrix {row: {col: score}} from the unordered pairing results."""
    table = {a: {} for a in names}
    for a, b, counts in pair_results:
        s = score_from_counts(counts)
        table[a][b] = s
        table[b][a] = 1 - s
    return table


def print_table(names, table):
    w = max(len(n) for n in names) + 1
    print(" " * w + "".join(f"{n[:10]:>11}" for n in names) + "      mean")
    for a in names:
        row = [table[a].get(b) for b in names]
        vals = [v for v in row if v is not None]
        cells = "".join(f"{'   -':>11}" if v is None else f"{v:11.2f}" for v in row)
        print(f"{a:<{w}}{cells}{sum(vals) / len(vals):10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+", help="name=checkpoint_dir, in training order")
    ap.add_argument("--games", type=int, default=30, help="games per colour per pairing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--start-file", type=Path, default=None, help="CSV with a `fen` column: games start from these positions instead of the opening")
    args = ap.parse_args()
    entries = [e.split("=", 1) for e in args.entries]
    names = [n for n, _ in entries]
    start_fens = None
    if args.start_file:
        from src.envs.start_positions import load_fens
        start_fens = load_fens(args.start_file)
    jobs = [(entries[i], entries[j], args.games, args.seed * 1000 + i * 31 + j, start_fens)
            for i in range(len(entries)) for j in range(i + 1, len(entries))]
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(play_pair, jobs))
    else:
        results = []
        for job in jobs:
            results.append(play_pair(job))
            print(f"  {results[-1][0]} vs {results[-1][1]}: {score_from_counts(results[-1][2]):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    table = build_table(names, results)
    print_table(names, table)
    print(f"{len(jobs)} pairings x {2 * args.games} games in {time.time() - t0:.0f}s")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"names": names, "dirs": dict(entries), "games_per_colour": args.games,
                                        "seed": args.seed, "table": table, "start_file": str(args.start_file) if args.start_file else None,
                                        "counts": [{"a": a, "b": b, "counts": c} for a, b, c in results]}, indent=1))


if __name__ == "__main__":
    main()
