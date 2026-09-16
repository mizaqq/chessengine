"""Profile where wall time goes in a training update (sync env).

Usage: python -m scripts.profile_update [--updates 20] [--puzzle-fraction 0.5]
Prints cumulative time for the pieces of one update: env stepping (OpenSpiel C++),
observation/mask building (Python glue), model forward, backward, returns.
"""
import argparse
import cProfile
import pstats
import io

from src.entrypoints.train import run_training_from_config

BUCKETS = {
    "whole training loop": ("training.py", "run_chess_training"),
    "  rollout collection (env + forward)": ("training.py", "_collect_rollout"),
    "    vector env step": ("open_spiel_vector_env.py", "step"),
    "      OpenSpiel env.step": ("open_spiel_env.py", "step"),
    "      env.state() observation glue": ("open_spiel_env.py", "state"),
    "      get_legal_actions() mask glue": ("open_spiel_env.py", "get_legal_actions"),
    "      env.reset (incl. FEN starts)": ("open_spiel_env.py", "reset"),
    "    model forward (rollout + bootstrap)": ("chess_model.py", "forward"),
    "  GAE walk": ("returns.py", "compute_gae_for_model"),
    "  update step (re-eval forward+backward+step)": ("training.py", "update_model"),
    "    backward": ("", "backward"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--puzzle-fraction", type=float, default=0.5)
    ap.add_argument("--env-type", default="sync")
    args = ap.parse_args()
    config = {
        "num_envs": 12, "max_updates": args.updates, "steps_per_update": 15, "seed": 0,
        "env_type": args.env_type, "puzzle_fraction": args.puzzle_fraction,
        "puzzle_train_file": "data/puzzles/mate_in_1_train.csv",
        "eval_interval": 10**9,
    }
    prof = cProfile.Profile()
    prof.enable()
    run_training_from_config(config)
    prof.disable()

    stats = pstats.Stats(prof)
    print(f"\n{args.env_type} env, {args.updates} updates, puzzle_fraction {args.puzzle_fraction}")
    print(f"{'bucket':45s} {'cum s':>8s} {'per update':>11s}")
    for name, (file_part, func) in BUCKETS.items():
        cum = 0.0
        for (filename, _line, fn), (_cc, _nc, _tt, ct, _callers) in stats.stats.items():
            if fn == func and file_part in filename:
                cum += ct
        print(f"{name:45s} {cum:8.2f} {cum / args.updates:11.3f}")


if __name__ == "__main__":
    main()
