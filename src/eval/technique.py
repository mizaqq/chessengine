"""Held-out technique evaluation: fixed generated bare-king starts per material set,
both sides sampling from the learner; `technique_rate_<label>` = share mated within
the cap. The starts come from a fixed seed, so every evaluation plays the same
positions (like the finishing evaluation's fixed records)."""
from typing import Dict, Sequence

import torch

from src.envs.start_positions import WHITE
from src.envs.technique import TechniqueSampler
from src.eval.finishes import play_finishes


def technique_starts(label: str, games: int, cap: int, seed: int):
    sampler = TechniqueSampler([label], {label: cap}, seed=seed)
    return [sampler.sample() for _ in range(games)]


def evaluate_technique(white_model, black_model, sets: Sequence[str], caps: Dict[str, int],
                       games: int = 50, oriented: bool = True, seed: int = 0) -> Dict[str, float]:
    models = {WHITE: white_model, 0: black_model}
    was_training = {pid: m.training for pid, m in models.items()}
    for m in models.values():
        m.eval()
    gen = torch.Generator().manual_seed(seed)
    out: Dict[str, float] = {}
    try:
        for k, label in enumerate(sets):
            starts = technique_starts(label, games, int(caps[label]), seed + 100 * k)
            results = play_finishes(white_model, black_model, [(s.fen, s.cap) for s in starts], oriented, gen)
            won = sum(res == ("white_win" if s.winner == WHITE else "black_win") for s, res in zip(starts, results))
            out[f"technique_rate_{label}"] = won / len(starts)
            out[f"technique_games_{label}"] = len(starts)
    finally:
        for pid, m in models.items():
            m.train(was_training[pid])
    return out
