"""Build the human-move pretraining set from the Lichess games database (CC0).

Streams one monthly PGN dump, keeps games where both players are rated at least
`--min-elo`, samples `--per-game` positions per game (AlphaGo's value network
overfit when every position of a game was used: successive positions are near
copies sharing one outcome), and encodes each position exactly as the engine sees
it: OpenSpiel observation oriented to the mover, legal mask, the human move as an
action id (via the SAN bridge in `OpenSpielEnv.actions_for_uci`), and the game
outcome from the mover's side (+1 win, -1 loss, -0.25 draw, matching the training
rewards divided by 2). Train and held-out are split by game.

Also writes mid-game start positions (one FEN per game at plies 20-60) for the
`midgame_boards` option.

Usage:
  python -m scripts.prepare_games [--month 2013-12] [--min-elo 1500] [--positions 315000]
                                  [--per-game 8] [--eval-frac 0.05] [--midgame 50000] [--seed 0]
Outputs: data/games/human_train.npz, data/games/human_eval.npz, data/games/midgame_starts.csv
"""
import argparse
import csv
import io
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
import zstandard

from src.envs.open_spiel_env import OpenSpielEnv
from src.model.orientation import orient_black

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/games")
URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
RESULT_WHITE = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": -0.25}
WHITE, BLACK = 1, 0
NO_PROGRESS_PLANE = 15


@dataclass
class Sample:
    fen: str
    move_uci: str
    mover: int          # 1 white, 0 black (OpenSpiel convention)
    z: float            # outcome from the mover's side
    game_index: int


def download(month: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"using existing {dest}")
        return dest
    url = URL.format(month=month)
    print(f"downloading {url} -> {dest}")
    subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "--fail", "-o", str(dest), url], check=True)
    return dest


def iter_games(handle):
    """Yield python-chess games from an open text handle until exhausted."""
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            return
        yield game


def open_pgn(path: Path):
    reader = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def open_pgn_stream(month: str):
    """Stream a Lichess month straight from the server (curl | zstd), so a 30 GB dump is read
    only as far as the position quota needs. Returns (text handle, curl process)."""
    proc = subprocess.Popen(["curl", "-sL", "--fail", "--retry", "3", URL.format(month=month)], stdout=subprocess.PIPE)
    reader = zstandard.ZstdDecompressor().stream_reader(proc.stdout)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), proc


_ELO = re.compile(r'^\[(White|Black)Elo "(\d+)"')


def iter_games_filtered(handle, min_elo: int):
    """Yield parsed games whose two Elo headers are both >= min_elo, skipping the movetext
    of every other game without parsing it (python-chess parsing is the bottleneck; on a
    recent month most games are below a 2000 floor)."""
    headers, elos, in_moves, moves = [], {}, False, []
    for line in handle:
        if line.startswith("["):
            if in_moves:               # new game starts: flush the previous one
                if len(elos) == 2 and min(elos.values()) >= min_elo:
                    g = chess.pgn.read_game(io.StringIO("".join(headers) + "\n" + "".join(moves)))
                    if g is not None:
                        yield g
                headers, elos, in_moves, moves = [], {}, False, []
            headers.append(line)
            m = _ELO.match(line)
            if m:
                elos[m.group(1)] = int(m.group(2))
        elif line.strip():
            in_moves = True
            if len(elos) == 2 and min(elos.values()) >= min_elo:
                moves.append(line)
        else:
            if headers and not in_moves:
                in_moves = True        # blank line after the headers
    if headers and len(elos) == 2 and min(elos.values()) >= min_elo:
        g = chess.pgn.read_game(io.StringIO("".join(headers) + "\n" + "".join(moves)))
        if g is not None:
            yield g


def game_passes(game, min_elo: int, min_plies: int = 12) -> bool:
    h = game.headers
    if h.get("Result") not in RESULT_WHITE:
        return False
    try:
        if int(h["WhiteElo"]) < min_elo or int(h["BlackElo"]) < min_elo:
            return False
    except (KeyError, ValueError):
        return False
    return True


def sample_game(game, game_index: int, per_game: int, rng: random.Random, min_plies: int = 12):
    """Return (samples, midgame_fen or None) for one game."""
    moves = list(game.mainline_moves())
    if len(moves) < min_plies:
        return [], None
    z_white = RESULT_WHITE[game.headers["Result"]]
    picks = set(rng.sample(range(len(moves)), min(per_game, len(moves))))
    mid_ply = rng.randint(20, min(60, len(moves) - 1)) if len(moves) > 20 else None
    board = game.board()
    out, mid_fen = [], None
    for ply, move in enumerate(moves):
        if ply == mid_ply:
            mid_fen = board.fen()
        if ply in picks:
            mover = WHITE if board.turn == chess.WHITE else BLACK
            z = z_white if mover == WHITE else -z_white
            out.append(Sample(board.fen(), move.uci(), mover, z, game_index))
        board.push(move)
    return out, mid_fen


def encode(samples, env: OpenSpielEnv | None = None):
    """Encode samples with OpenSpiel. Returns dict of arrays; skips positions whose
    move cannot be mapped to an action id."""
    env = env or OpenSpielEnv()
    obs_bits, mask_bits, actions, zs, movers, nops, kept = [], [], [], [], [], [], 0
    for s in samples:
        env.reset(s.fen)
        ids = env.actions_for_uci([s.move_uci])
        if len(ids) != 1:
            continue
        obs = torch.tensor(env.state(), dtype=torch.float32)
        if s.mover == BLACK:
            obs = orient_black(obs)
        # Plane 15 is the no-progress counter as one fraction repeated over the
        # board; every other plane is 0/1. Bit-pack the planes, keep the fraction.
        arr = obs.numpy()
        nops.append(float(arr[NO_PROGRESS_PLANE, 0, 0]))
        arr = arr.copy(); arr[NO_PROGRESS_PLANE] = 0
        obs_bits.append(np.packbits(arr.astype(bool).ravel()))
        mask_bits.append(np.packbits(env.get_legal_actions().numpy().astype(bool)))
        actions.append(next(iter(ids)))
        zs.append(s.z)
        movers.append(s.mover)
        kept += 1
    return {
        "obs": np.stack(obs_bits) if kept else np.zeros((0, 160), np.uint8),
        "nop": np.array(nops, dtype=np.float32),
        "mask": np.stack(mask_bits) if kept else np.zeros((0, 585), np.uint8),
        "action": np.array(actions, dtype=np.int16),
        "z": np.array(zs, dtype=np.float32),
        "mover": np.array(movers, dtype=np.int8),
    }


def unpack_obs(obs_bits: np.ndarray, nop: np.ndarray) -> torch.Tensor:
    obs = torch.tensor(np.unpackbits(obs_bits, axis=1)[:, : 20 * 64].reshape(-1, 20, 8, 8), dtype=torch.float32)
    obs[:, NO_PROGRESS_PLANE] = torch.tensor(nop, dtype=torch.float32)[:, None, None]
    return obs


def unpack_mask(mask_bits: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.unpackbits(mask_bits, axis=1)[:, :4674], dtype=torch.float32)


def collect(handle, min_elo, positions, per_game, eval_frac, midgame, seed, log_every=20_000, prefilter=False):
    rng = random.Random(seed)
    train, ev, mid_fens = [], [], []
    seen = kept_games = 0
    games = iter_games_filtered(handle, min_elo) if prefilter else iter_games(handle)
    for game in games:
        seen += 1
        if seen % log_every == 0:
            print(f"  games {seen}, kept {kept_games}, positions {len(train) + len(ev)}", flush=True)
        if not game_passes(game, min_elo):
            continue
        samples, mid = sample_game(game, kept_games, per_game, rng)
        if not samples:
            continue
        kept_games += 1
        (ev if rng.random() < eval_frac else train).extend(samples)
        if mid is not None and len(mid_fens) < midgame:
            mid_fens.append(mid)
        if len(train) + len(ev) >= positions:
            break
    print(f"scanned {seen} games, kept {kept_games}; {len(train)} train / {len(ev)} eval positions, {len(mid_fens)} mid-game FENs")
    return train, ev, mid_fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2013-12")
    ap.add_argument("--min-elo", type=int, default=1500)
    ap.add_argument("--positions", type=int, default=315_000)
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--midgame", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="human", help="output files <prefix>_train.npz / <prefix>_eval.npz")
    ap.add_argument("--midgame-out", default="midgame_starts.csv")
    ap.add_argument("--stream", action="store_true", help="read the month from the server without saving it; skips sub-floor games before parsing")
    args = ap.parse_args()

    if args.stream:
        handle, proc = open_pgn_stream(args.month)
        train, ev, mid_fens = collect(handle, args.min_elo, args.positions, args.per_game,
                                      args.eval_frac, args.midgame, args.seed, prefilter=True)
        proc.kill()
    else:
        dump = download(args.month, RAW_DIR / f"lichess_db_standard_rated_{args.month}.pgn.zst")
        train, ev, mid_fens = collect(open_pgn(dump), args.min_elo, args.positions, args.per_game,
                                      args.eval_frac, args.midgame, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / args.midgame_out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fen"]); w.writerows([[f] for f in mid_fens])
    env = OpenSpielEnv()
    for name, samples in (("eval", ev), ("train", train)):
        print(f"encoding {name} ({len(samples)} positions)", flush=True)
        arrays = encode(samples, env)
        out = OUT_DIR / f"{args.prefix}_{name}.npz"
        np.savez_compressed(out, **arrays)
        print(f"  wrote {len(arrays['action'])} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
