import torch
import numpy as np
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

    def reset(self):
        return self.env.reset()

    def step(self, action):
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
