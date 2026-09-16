import torch
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import chess


class OpenSpielEnv:
    """Single OpenSpiel chess environment wrapper."""
    
    def __init__(self):
        self.env = rl_environment.Environment("chess")
        self.env.reset()
        self.rewards_dict = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

    def reset(self, fen: str | None = None):
        """Start a new game from the opening, or from `fen` when given.

        OpenSpiel's chess game has no FEN parameter, so a custom start is built
        with `new_initial_state(fen)` and installed with `set_state`.
        """
        time_step = self.env.reset()
        if fen is None:
            return time_step
        game = pyspiel.load_game("chess")
        self.env.set_state(game.new_initial_state(fen))
        return self.env.get_time_step()

    def step(self, action):
        if not isinstance(action, list):
            action = [action]
        return self.env.step(action)

    def get_current_player(self):
        return self.env.get_time_step().current_player()

    def state(self):
        current_player = self.get_current_player()
        if current_player != 0 and current_player != 1:
            current_player = 1 if self.env.get_time_step().rewards[0] > 0 else 0
        return np.array(
            self.env.get_time_step().observations["info_state"][current_player]
        ).reshape(20, 8, 8)

    def get_legal_actions(self):
        current_player = self.get_current_player()
        if current_player != 0 and current_player != 1:
            current_player = 1 if self.env.get_time_step().rewards[0] > 0 else 0
        legal_actions = self.env.get_time_step().observations["legal_actions"][
            current_player
        ]
        mask = torch.zeros(4674)
        mask[legal_actions] = 1.0
        return mask

    def is_done(self):
        return self.env.get_time_step().last()

    def game_result(self):
        """Return game result string or None if game is not over."""
        if not self.is_done():
            return None
        rewards = self.env.get_time_step().rewards
        if rewards[1] > 0:
            return "white_win"
        elif rewards[0] > 0:
            return "black_win"
        return "draw"
