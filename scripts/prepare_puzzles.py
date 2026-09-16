"""Build the mate-in-one puzzle sets from the Lichess puzzle database (CC0).

Lichess stores the FEN *before* the opponent's move; the solution starts with the
second move. We apply the first move so the stored FEN is the position where the
side to move has a mate in one, and store every checkmating move (UCI) from there.

Depth 2 (`--depth 2`, theme mateIn2): the stored FEN is after the opponent's first
move; `key_moves` is the puzzle's forcing move, kept only if a rules-engine check
confirms every reply leaves a mate in one.

Usage: python -m scripts.prepare_puzzles [--depth 1|2] [--train 20000] [--eval 2000] [--seed 0]
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
FIELDS = ["puzzle_id", "fen", "key_moves", "mate_in", "rating"]


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 300_000_000:
        print(f"using existing {dest} (delete it to re-download)")
        return dest
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


def forces_mate_in_two(board: chess.Board, key_uci: str) -> bool:
    """True if `key_uci` leaves the opponent no reply that avoids mate on the move after."""
    board.push_uci(key_uci)
    try:
        if board.is_game_over():
            return False
        for reply in board.legal_moves:
            board.push(reply)
            has_mate = bool(mating_moves(board))
            board.pop()
            if not has_mate:
                return False
        return True
    finally:
        board.pop()


def convert_row(row: dict, depth: int = 1) -> dict | None:
    """Return a stored row for a mateIn<depth> puzzle, or None if it does not check out."""
    if f"mateIn{depth}" not in row["Themes"].split():
        return None
    moves = row["Moves"].split()
    if len(moves) != 2 * depth:
        return None
    board = chess.Board(row["FEN"])
    board.push_uci(moves[0])
    if depth == 1:
        key = mating_moves(board)
        if moves[1] not in key:
            return None
    else:
        if not forces_mate_in_two(board, moves[1]):
            return None
        key = [moves[1]]
    return {
        "puzzle_id": row["PuzzleId"],
        "fen": board.fen(),
        "key_moves": " ".join(key),
        "mate_in": str(depth),
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
    ap.add_argument("--depth", type=int, default=1, choices=(1, 2))
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", type=Path, default=RAW_DIR / "lichess_db_puzzle.csv.zst")
    ap.add_argument("--keep-eval", type=Path, default=None,
                    help="existing held-out file to keep unchanged; its ids are excluded from train")
    args = ap.parse_args()

    download(DUMP_URL, args.dump)
    rows = []
    seen = 0
    for raw in iter_dump(args.dump):
        seen += 1
        converted = convert_row(raw, args.depth)
        if converted is not None:
            rows.append(converted)
        if seen % 500_000 == 0:
            print(f"  scanned {seen}, kept {len(rows)}", flush=True)
    print(f"scanned {seen} puzzles, kept {len(rows)} mate-in-{args.depth}")
    prefix = f"mate_in_{args.depth}"

    random.Random(args.seed).shuffle(rows)
    if args.keep_eval:
        # Same seed and shuffle as before: the held-out rows sit at positions
        # [old_train, old_train + eval) of this order. Drop them by id and take the
        # first `train` of the rest, so the old train rows stay a prefix.
        with open(args.keep_eval, newline="") as fh:
            keep_ids = {r["puzzle_id"] for r in csv.DictReader(fh)}
        rows = [r for r in rows if r["puzzle_id"] not in keep_ids]
        if len(rows) < args.train:
            raise SystemExit(f"only {len(rows)} usable puzzles, need {args.train}")
        write_csv(OUT_DIR / f"{prefix}_train.csv", rows[: args.train])
        print(f"wrote {args.train} train rows to {OUT_DIR}; held-out {args.keep_eval} unchanged")
        return
    need = args.train + args.eval
    if len(rows) < need:
        raise SystemExit(f"only {len(rows)} usable puzzles, need {need}")
    write_csv(OUT_DIR / f"{prefix}_train.csv", rows[: args.train])
    write_csv(OUT_DIR / f"{prefix}_eval.csv", rows[args.train : need])
    print(f"wrote {args.train} train and {args.eval} eval rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
