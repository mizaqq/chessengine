"""Summarise the add-ppo-gae arms: table of final metrics + curves PNG.

Usage: python -m scripts.summarize_ppo_arms [experiments/ppo-gae]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def last_with(logs, key):
    for e in reversed(logs):
        if e.get(key) is not None:
            return e[key]
    return None


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/ppo-gae")
    arms = sorted(p for p in root.iterdir() if (p / "run.json").exists())
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("mean_entropy_normalized_game", "opening-board entropy"),
        ("mean_entropy_normalized_puzzle", "puzzle-board entropy"),
        ("puzzle_solved_rate", "puzzle solved rate (train)"),
        ("mean_value_loss", "value loss"),
        ("mean_clip_fraction", "clip fraction (ppo)"),
        ("mean_approx_kl", "approx KL"),
    ]
    for arm in arms:
        run = json.loads((arm / "run.json").read_text())
        logs = run["logs"]
        games = json.loads((arm / "games.json").read_text()) if (arm / "games.json").exists() else {}
        rows.append({
            "arm": arm.name,
            "s/update": run["elapsed_s"] / run["config"]["max_updates"],
            "entropy_game": last_with(logs, "mean_entropy_normalized_game"),
            "puzzle_solved": last_with(logs, "puzzle_solved_rate"),
            "lichess_top1": last_with(logs, "lichess_top1"),
            "selfplay_top1": last_with(logs, "selfplay_top1"),
            "draw_rate": last_with(logs, "draw_rate"),
            "clip_frac": last_with(logs, "mean_clip_fraction"),
            "approx_kl": last_with(logs, "mean_approx_kl"),
            "in_game_mate": games.get("mate_in_one_rate"),
            "opps": games.get("mate_in_one_opportunities"),
            "mean_plies": games.get("mean_plies"),
        })
        for ax, (key, title) in zip(axes.flat, panels):
            xs = [e["episode"] for e in logs if e.get(key) is not None]
            ys = [e[key] for e in logs if e.get(key) is not None]
            if xs:
                ax.plot(xs, ys, label=arm.name)
            ax.set_title(title)
    for ax in axes.flat:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "summary.png", dpi=110)
    keys = list(rows[0].keys())
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join("" if r[k] is None else (f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])) for k in keys))
    (root / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
