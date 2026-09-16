"""Build the mate-in-one puzzle sets from the Lichess puzzle database (CC0).

Lichess stores the FEN *before* the opponent's move; the solution starts with the
second move. We apply the first move so the stored FEN is the position where the
side to move has a mate in one, and store every checkmating move (UCI) from there.

Usage: python scripts/prepare_puzzles.py [--train 20000] [--eval 2000] [--seed 0]
"""
import argparse
import csv
import io
import random
import subprocess
from pathlib import Path

import chess
import zstandard

DUMP_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/puzzles")
FIELDS = ["puzzle_id", "fen", "mating_moves", "rating"]


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # curl resumes a partial file (-C -) and retries; the dump is ~300 MB.
    print(f"downloading {url} -> {dest} (resumes if partial)")
    subprocess.run(
        ["curl", "-L", "-C", "-", "--retry", "5", "--fail", "-o", str(dest), url],
        check=True,
    )
    return dest


def mating_moves(board: chess.Board) -> list[str]:
    out = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            out.append(move.uci())
        board.pop()
    return out


def convert_row(row: dict) -> dict | None:
    """Return a stored row for a mateIn1 puzzle, or None if it does not check out."""
    if "mateIn1" not in row["Themes"].split():
        return None
    moves = row["Moves"].split()
    if len(moves) != 2:
        return None
    board = chess.Board(row["FEN"])
    board.push_uci(moves[0])
    mates = mating_moves(board)
    if moves[1] not in mates:
        return None
    return {
        "puzzle_id": row["PuzzleId"],
        "fen": board.fen(),
        "mating_moves": " ".join(mates),
        "rating": row["Rating"],
    }


def iter_dump(path: Path):
    with open(path, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        yield from csv.DictReader(text)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", type=Path, default=RAW_DIR / "lichess_db_puzzle.csv.zst")
    args = ap.parse_args()

    download(DUMP_URL, args.dump)
    rows = []
    seen = 0
    for raw in iter_dump(args.dump):
        seen += 1
        converted = convert_row(raw)
        if converted is not None:
            rows.append(converted)
    print(f"scanned {seen} puzzles, kept {len(rows)} mate-in-one")

    random.Random(args.seed).shuffle(rows)
    need = args.train + args.eval
    if len(rows) < need:
        raise SystemExit(f"only {len(rows)} usable puzzles, need {need}")
    write_csv(OUT_DIR / "mate_in_1_train.csv", rows[: args.train])
    write_csv(OUT_DIR / "mate_in_1_eval.csv", rows[args.train : need])
    print(f"wrote {args.train} train and {args.eval} eval rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
