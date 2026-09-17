"""Held-out finishing evaluation: conversion rate from rewound human mates.

For each depth k, up to `games` held-out records that support k are rewound k
plies before the mating move (winner to move) and played out with the same
network on both sides, moves sampled from the policy, under a cap of
k + cap_margin plies (draw at the cap). Reported: `finish_rate_d<k>`, the share
of games the record's winner won, and `finish_games_d<k>`.
"""
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.start_positions import FinishRecord
from src.model.orientation import orient

WHITE = 1
BLACK = 0


@torch.no_grad()
def play_finishes(white_model, black_model, starts, oriented: bool, generator) -> List[str]:
    """Play every (fen, cap) start in lockstep; return the result string per start."""
    envs = [OpenSpielEnv() for _ in starts]
    for env, (fen, cap) in zip(envs, starts):
        env.reset(fen, max_plies=cap)
    models = {WHITE: white_model, BLACK: black_model}
    while True:
        active = [i for i, e in enumerate(envs) if not e.is_done()]
        if not active:
            break
        obs = torch.tensor(np.array([envs[i].state() for i in active]), dtype=torch.float32)
        legal = torch.stack([envs[i].get_legal_actions() for i in active])
        players = torch.tensor([envs[i].get_current_player() for i in active])
        obs_in = orient(obs, players) if oriented else obs
        actions = torch.zeros(len(active), dtype=torch.long)
        for pid, model in models.items():
            m = players == pid
            if m.any():
                probs, _ = model(obs_in[m], legal[m])
                actions[m] = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        for j, i in enumerate(active):
            envs[i].step(int(actions[j]))
    return [e.game_result() for e in envs]


def evaluate_finishes(
    white_model: torch.nn.Module,
    black_model: torch.nn.Module,
    records: List[FinishRecord],
    depths: Sequence[int] = (2, 10, 20),
    games: int = 100,
    cap_margin: int = 20,
    oriented: bool = False,
    seed: int = 0,
) -> Dict[str, float]:
    models = {WHITE: white_model, BLACK: black_model}
    was_training = {pid: m.training for pid, m in models.items()}
    for m in models.values():
        m.eval()
    gen = torch.Generator().manual_seed(seed)
    out: Dict[str, float] = {}
    try:
        for depth in depths:
            usable = [r for r in records if r.max_depth() >= depth][:games]
            starts = [(r.rewind(depth), depth + cap_margin) for r in usable]
            results = play_finishes(white_model, black_model, starts, oriented, gen) if starts else []
            won = sum(
                res == ("white_win" if r.winner == WHITE else "black_win")
                for r, res in zip(usable, results)
            )
            out[f"finish_rate_d{depth}"] = won / len(usable) if usable else None
            out[f"finish_games_d{depth}"] = len(usable)
    finally:
        for pid, m in models.items():
            m.train(was_training[pid])
    return out
