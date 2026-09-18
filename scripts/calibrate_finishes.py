"""Calibrate the finishing-conversion dial: how convertible are the rewound human
positions at all, against OUR sampling defender?

Attackers on the same held-out starts per depth:
  sampled   : the checkpoint samples (the dial as reported in training)
  greedy    : the checkpoint plays argmax
  human     : the recorded human moves while the defender stays on the record, then the
              checkpoint greedy once the defender has deviated
The human attacker is the ceiling we have been implicitly chasing: if even the human's
own line converts rarely against our defender, the mate depended on the human defender's
blunders and the dial is saturated (owner 2026-09-19: "run a calibration").

Usage: python -m scripts.calibrate_finishes <ckpt_dir> [--depths 2 10 20] [--games 100] [--out calib.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.start_positions import WHITE, BLACK, load_finishes
from src.model.checkpoints import load_models
from src.model.orientation import orient


def play(model, records, depth, cap_margin, mode, gen):
    starts = [(r.rewind(depth), r.moves[len(r.moves) - 1 - depth:], r.winner) for r in records]
    envs = [OpenSpielEnv() for _ in starts]
    for env, (fen, _, _) in zip(envs, starts):
        env.reset(fen, max_plies=depth + cap_margin)
    on_line = [mode == "human"] * len(starts)
    ply = [0] * len(starts)
    left_line_at = [None] * len(starts)
    while True:
        active = [i for i, e in enumerate(envs) if not e.is_done()]
        if not active:
            break
        obs = torch.tensor(np.array([envs[i].state() for i in active]), dtype=torch.float32)
        legal = torch.stack([envs[i].get_legal_actions() for i in active])
        players = torch.tensor([envs[i].get_current_player() for i in active])
        with torch.no_grad():
            probs, _ = model(orient(obs, players), legal)
        sampled = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        greedy = probs.argmax(dim=1)
        for j, i in enumerate(active):
            fen, line, winner = starts[i]
            is_attacker = int(players[j]) == winner
            action = None
            if is_attacker:
                if mode == "human" and on_line[i] and ply[i] < len(line):
                    ids = envs[i].actions_for_uci([line[ply[i]]])
                    action = next(iter(ids)) if ids else None
                if action is None:
                    action = int(greedy[j]) if mode in ("greedy", "human") else int(sampled[j])
            else:
                action = int(sampled[j])
                if on_line[i] and ply[i] < len(line):
                    ids = envs[i].actions_for_uci([line[ply[i]]])
                    if action not in ids:
                        on_line[i] = False
                        left_line_at[i] = ply[i]
            envs[i].step(action)
            ply[i] += 1
    results = [e.game_result() for e in envs]
    won = sum(res == ("white_win" if w == WHITE else "black_win") for res, (_, _, w) in zip(results, starts))
    out = {"rate": won / len(starts), "games": len(starts)}
    if mode == "human":
        stayed = sum(1 for x in on_line if x)
        out["defender_stayed_on_line"] = stayed / len(starts)
        lefts = [x for x in left_line_at if x is not None]
        out["mean_ply_defender_left_line"] = float(np.mean(lefts)) if lefts else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--eval", default="data/finishes/finish_eval.csv")
    ap.add_argument("--depths", type=int, nargs="*", default=[2, 10, 20])
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--cap-margin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    model, _, oriented, paths = load_models(args.ckpt_dir)
    assert oriented
    model.eval()
    records = load_finishes(args.eval)
    out = {"checkpoint": paths, "depths": {}}
    for depth in args.depths:
        usable = [r for r in records if r.max_depth() >= depth][: args.games]
        row = {}
        for mode in ("sampled", "greedy", "human"):
            gen = torch.Generator().manual_seed(args.seed)
            row[mode] = play(model, usable, depth, args.cap_margin, mode, gen)
        out["depths"][str(depth)] = row
        print(f"depth {depth:2d}: sampled {row['sampled']['rate']:.2f}  greedy {row['greedy']['rate']:.2f}  "
              f"human-line {row['human']['rate']:.2f}  (defender stayed on the record {row['human']['defender_stayed_on_line']:.2f}, "
              f"left at ply {row['human']['mean_ply_defender_left_line']})", flush=True)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
