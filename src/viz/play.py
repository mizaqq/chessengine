"""Play one game between two policy models and record what they thought.

Used by the visualisation notebook. Player convention follows OpenSpiel:
player 1 = white, player 0 = black.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.open_spiel_vector_env import _material_np
from src.model.orientation import orient_black

WHITE = 1
BLACK = 0


@dataclass
class MoveRecord:
    move_number: int
    side: str                     # "white" | "black"
    san: str
    fen_before: str
    fen_after: str
    value: float                  # mover's value head output at fen_before
    material: float               # material from the mover's perspective at fen_before
    value_plus_material: float    # value with the shaping offset removed (V' + Phi)
    top_moves: List[Tuple[str, float]]  # up to 3 (san, probability)
    prob_played: float


@dataclass
class GameRecord:
    moves: List[MoveRecord] = field(default_factory=list)
    result: str = "unfinished"    # "white_win" | "black_win" | "draw" | "unfinished"
    final_fen: str = ""


def play_game(
    white_model: torch.nn.Module,
    black_model: torch.nn.Module,
    greedy: bool = False,
    max_moves: int = 300,
    seed: Optional[int] = None,
    greedy_white: Optional[bool] = None,
    greedy_black: Optional[bool] = None,
    oriented: bool = False,
) -> GameRecord:
    """Play a single game and return per-move records plus the result.

    `greedy` applies to both sides; `greedy_white` / `greedy_black` override it per
    side. Greedy means argmax (deterministic, ignores `seed`); otherwise moves are
    sampled from the policy.
    """
    greedy_by_player = {
        WHITE: greedy if greedy_white is None else greedy_white,
        BLACK: greedy if greedy_black is None else greedy_black,
    }
    if seed is not None:
        torch.manual_seed(seed)
    env = OpenSpielEnv()
    env.reset()
    models = {WHITE: white_model.eval(), BLACK: black_model.eval()}
    game = GameRecord()

    for move_number in range(1, max_moves + 1):
        if env.is_done():
            break
        state = env.env.get_state
        player = env.get_current_player()
        fen_before = state.to_string()
        obs = torch.tensor(env.state(), dtype=torch.float32).unsqueeze(0)
        if oriented and player == BLACK:
            obs = orient_black(obs)
        mask = env.get_legal_actions().unsqueeze(0)

        with torch.no_grad():
            probs, value = models[player](obs, mask)
        probs = probs.squeeze(0)
        action = int(probs.argmax()) if greedy_by_player[player] else int(torch.multinomial(probs, 1))

        white_material = _material_np(env.state())
        material = white_material if player == WHITE else -white_material
        top_idx = torch.topk(probs, k=min(3, int(mask.sum()))).indices.tolist()
        top_moves = [(state.action_to_string(player, a), float(probs[a])) for a in top_idx]
        san = state.action_to_string(player, action)

        env.step([action])
        game.moves.append(
            MoveRecord(
                move_number=move_number,
                side="white" if player == WHITE else "black",
                san=san,
                fen_before=fen_before,
                fen_after=env.env.get_state.to_string(),
                value=float(value.item()),
                material=float(material),
                value_plus_material=float(value.item() + material),
                top_moves=top_moves,
                prob_played=float(probs[action]),
            )
        )

    game.final_fen = env.env.get_state.to_string()
    game.result = env.game_result() or "unfinished"
    return game
