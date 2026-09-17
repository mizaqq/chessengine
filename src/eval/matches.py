"""Sampled matches between two policies, both colours (shared-network checkpoints).

Score from the first player's side: win 1, draw or 300-move cap 0.5, loss 0.
Used by scripts/eval_matrix.py (win matrix, Bansal et al. 2017 Table 1) and by
the periodic `prior_score` evaluation.
"""
from collections import Counter
from typing import Dict

import torch

from src.viz.play import play_game

RESULT_SCORE_WHITE = {"white_win": 1.0, "black_win": 0.0, "draw": 0.5, "unfinished": 0.5}


def play_match(model_a, model_b, games_per_colour: int, seed: int = 0) -> Dict[str, int]:
    """Counts keyed `a_<colour>_<result>`, colour = the colour model_a played."""
    counts: Counter = Counter()
    torch.manual_seed(seed)
    for _ in range(games_per_colour):
        counts["a_white_" + play_game(model_a, model_b, greedy=False, oriented=True).result] += 1
        counts["a_black_" + play_game(model_b, model_a, greedy=False, oriented=True).result] += 1
    return dict(counts)


def score_from_counts(counts) -> float:
    total = points = 0
    for key, n in counts.items():
        colour, result = key.split("_", 2)[1], key.split("_", 2)[2]
        s = RESULT_SCORE_WHITE[result]
        points += n * (s if colour == "white" else 1 - s)
        total += n
    return points / total if total else None


def summarize_match(counts) -> Dict[str, float]:
    wins = counts.get("a_white_white_win", 0) + counts.get("a_black_black_win", 0)
    losses = counts.get("a_white_black_win", 0) + counts.get("a_black_white_win", 0)
    return {"score": score_from_counts(counts), "wins": wins, "losses": losses, "games": sum(counts.values())}
