"""Probe a checkpoint on rewound human mates: policy mass on the mating move(s) and
the value head's prediction from the mover's view, at depths 0, 2 and 10; plus the
value on Lichess mate-in-one puzzles and own-game mate-in-one positions.

Usage: python -m scripts.probe_finish_values <ckpt_dir> [<ckpt_dir> ...] [--n 300]
"""
import argparse

import numpy as np
import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.start_positions import WHITE, load_finishes, load_puzzles
from src.eval.puzzles import mating_actions
from src.model.checkpoints import load_models
from src.model.orientation import orient_black


@torch.no_grad()
def probe(model, fens, with_mate=False):
    env = OpenSpielEnv()
    vals, mass, top1 = [], [], []
    for fen in fens:
        env.reset(fen)
        obs = torch.tensor(env.state(), dtype=torch.float32)[None]
        mask = env.get_legal_actions()[None]
        if env.get_current_player() != WHITE:
            obs = orient_black(obs)
        p, v = model(obs, mask)
        vals.append(v.item())
        if with_mate:
            m = mating_actions(env)
            mass.append(p[0, m].sum().item())
            top1.append(int(p[0].argmax()) in m)
    out = {"value": round(float(np.mean(vals)), 3)}
    if with_mate:
        out.update(mate_mass=round(float(np.mean(mass)), 3), top1=round(float(np.mean(top1)), 3))
        m = np.array(mass)
        out["hist"] = {f"<{hi}": round(float(((m >= lo) & (m < hi)).mean()), 3)
                       for lo, hi in ((0, 0.05), (0.05, 0.5), (0.5, 0.8), (0.8, 1.01))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    recs = [r for r in load_finishes("data/finishes/finish_eval.csv") if r.max_depth() >= 10][: args.n]
    lichess = [p.fen for p in load_puzzles("data/puzzles/mate_in_1_eval.csv")[: args.n]]
    own = [p.fen for p in load_puzzles("data/puzzles/selfplay_mate_in_1_eval.csv")[: args.n]]
    for ck in args.ckpts:
        w, _, _, _ = load_models(ck)
        w.eval()
        print(ck)
        print("  human mate d0 :", probe(w, [r.rewind(0) for r in recs], True))
        print("  human mate d2 :", probe(w, [r.rewind(2) for r in recs]))
        print("  human mate d10:", probe(w, [r.rewind(10) for r in recs]))
        print("  lichess m1    :", probe(w, lichess, True))
        print("  own-game m1   :", probe(w, own, True))


if __name__ == "__main__":
    main()
