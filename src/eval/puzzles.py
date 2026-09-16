"""Held-out mate-in-one evaluation.

For every puzzle position the model of the side to move is queried with the legal
move mask. Reported: top-1 accuracy (most probable move is a mate) and the mean
total probability placed on mating moves. Mating moves are computed from the board
(a child state that is terminal with a win for the mover), not read from the file,
so the metric does not depend on the UCI-to-action mapping.
"""
from typing import Dict, List

import numpy as np
import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.start_positions import Puzzle
from src.model.orientation import orient_black

WHITE = 1
BLACK = 0


def mating_actions(env: OpenSpielEnv) -> List[int]:
    state = env.env.get_state
    player = state.current_player()
    out = []
    for action in state.legal_actions():
        child = state.child(action)
        if child.is_terminal() and child.returns()[player] > 0:
            out.append(action)
    return out


def _positions(puzzles: List[Puzzle]):
    env = OpenSpielEnv()
    for puzzle in puzzles:
        env.reset(puzzle.fen)
        yield env.state(), env.get_legal_actions(), env.get_current_player(), mating_actions(env)


@torch.no_grad()
def evaluate_mate_in_one(
    white_model: torch.nn.Module,
    black_model: torch.nn.Module,
    puzzles: List[Puzzle],
    batch_size: int = 256,
    oriented: bool = False,
) -> Dict[str, float]:
    """Return {"puzzle_top1": ..., "puzzle_mate_prob": ..., "puzzle_count": n}.

    `oriented`: shared network; black-to-move observations are flipped to the
    mover's view before the forward pass."""
    models = {WHITE: white_model, BLACK: black_model}
    was_training = {pid: m.training for pid, m in models.items()}
    for m in models.values():
        m.eval()

    top1_hits = 0
    mate_prob_sum = 0.0
    per_player = {WHITE: [], BLACK: []}
    for obs, legal, player, mates in _positions(puzzles):
        per_player[player].append((obs, legal, mates))

    for player, items in per_player.items():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            obs = torch.tensor(np.stack([c[0] for c in chunk]), dtype=torch.float32)
            if oriented and player == BLACK:
                obs = orient_black(obs)
            legal = torch.stack([c[1] for c in chunk])
            probs, _ = models[player](obs, legal)
            top = probs.argmax(dim=1)
            for row, (_, _, mates) in enumerate(chunk):
                if not mates:
                    continue
                mate_idx = torch.tensor(mates)
                mate_prob_sum += probs[row, mate_idx].sum().item()
                top1_hits += int(top[row].item() in mates)

    for pid, m in models.items():
        m.train(was_training[pid])

    n = len(puzzles)
    return {
        "puzzle_top1": top1_hits / n if n else 0.0,
        "puzzle_mate_prob": mate_prob_sum / n if n else 0.0,
        "puzzle_count": n,
    }
