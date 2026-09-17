"""Build finishing records from human games that end in checkmate (Lichess DB, CC0).

Reverse curriculum data (Florensa et al. 2017): a finishing board starts k plies
before a human's mating move with the winner to move and must win under a ply
cap. Each record stores an anchor position at most `--window` plies before the
end, the moves from there up to and including the mate, and the winner; the
sampler rewinds to any even depth from one row. Only games where both players
are rated at least `--min-elo` are kept; train / held-out are split by game.

Usage: python -m scripts.prepare_finishes [--month 2013-12] [--min-elo 1500]
         [--train 20000] [--eval 2000] [--window 40] [--seed 0]
Outputs: data/finishes/finish_train.csv, data/finishes/finish_eval.csv
"""
import argparse
import csv
import random
from pathlib import Path

import chess
import chess.pgn

from scripts.prepare_games import RAW_DIR, download, game_passes, iter_games, open_pgn
from src.envs.start_positions import BLACK, WHITE, FinishRecord

OUT_DIR = Path("data/finishes")
FIELDS = ["game_id", "anchor_fen", "moves", "winner"]


def finish_record(game, game_id: str, window: int = 40):
    """FinishRecord for a game that ends in checkmate, else None."""
    moves = list(game.mainline_moves())
    board = game.board()
    for m in moves:
        board.push(m)
    if not board.is_checkmate():
        return None
    winner = WHITE if board.turn == chess.BLACK else BLACK   # the side to move is mated
    start = max(0, len(moves) - window)
    anchor = game.board()
    for m in moves[:start]:
        anchor.push(m)
    return FinishRecord(game_id, anchor.fen(), [m.uci() for m in moves[start:]], winner)


def collect(handle, min_elo: int, total: int, window: int, log_every: int = 20_000):
    records, seen = [], 0
    for game in iter_games(handle):
        seen += 1
        if seen % log_every == 0:
            print(f"  games {seen}, mates kept {len(records)}", flush=True)
        if not game_passes(game, min_elo):
            continue
        rec = finish_record(game, f"g{seen}", window)
        if rec is not None and len(rec.moves) >= 2:
            records.append(rec)
        if len(records) >= total:
            break
    print(f"scanned {seen} games, kept {len(records)} checkmate games")
    return records


def split(records, n_eval: int, seed: int):
    records = list(records)
    random.Random(seed).shuffle(records)
    return records[n_eval:], records[:n_eval]


def write_csv(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for r in records:
            w.writerow([r.game_id, r.anchor_fen, " ".join(r.moves), "white" if r.winner == WHITE else "black"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2013-12")
    ap.add_argument("--min-elo", type=int, default=1500)
    ap.add_argument("--train", type=int, default=20_000)
    ap.add_argument("--eval", type=int, default=2_000)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dump = download(args.month, RAW_DIR / f"lichess_db_standard_rated_{args.month}.pgn.zst")
    records = collect(open_pgn(dump), args.min_elo, args.train + args.eval, args.window)
    train, ev = split(records, args.eval, args.seed)
    write_csv(OUT_DIR / "finish_train.csv", train)
    write_csv(OUT_DIR / "finish_eval.csv", ev)
    print(f"wrote {len(train)} train / {len(ev)} held-out records to {OUT_DIR}")


if __name__ == "__main__":
    main()
