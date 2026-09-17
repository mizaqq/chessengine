import torch
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import chess


def _leaves_mate_in_one(board: chess.Board) -> bool:
    for m in board.legal_moves:
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def forcing_mate_actions_of(env) -> set[int]:
    """Legal actions after which every reply leaves a mate in one (a forced mate in
    two, checked with python-chess); empty if none. Demonstrator fallback."""
    state = env.env.get_state
    if state.current_player() not in (0, 1):
        return set()
    board = chess.Board(state.to_string())
    ucis = []
    for move in list(board.legal_moves):
        board.push(move)
        forced = not board.is_game_over()
        if forced:
            for reply in list(board.legal_moves):
                board.push(reply)
                ok = _leaves_mate_in_one(board)
                board.pop()
                if not ok:
                    forced = False
                    break
        board.pop()
        if forced:
            ucis.append(move.uci())
    return env.actions_for_uci(ucis) if ucis else set()


def mating_actions_of(env) -> list[int]:
    """Legal actions that mate at once (rules engine); [] if none."""
    state = env.env.get_state
    player = state.current_player()
    if player not in (0, 1):
        return []
    return [a for a in state.legal_actions() if state.child(a).is_terminal() and state.child(a).returns()[player] > 0]


class OpenSpielEnv:
    """Single OpenSpiel chess environment wrapper."""
    
    def __init__(self):
        self.env = rl_environment.Environment("chess")
        self.env.reset()
        self.puzzle_moves = 0   # >0: puzzle episode, ends after the mover's n-th move
        self.key_actions = None # puzzle_moves > 1: the mover's first move must be one of these
        self.max_plies = None   # game boards: end as a draw at this ply count
        self.plies = 0
        self.mover = None
        self.mover_moves = 0
        self.failed = False
        self.rewards_dict = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

    def set_demo_line(self, ucis) -> None:
        """Demonstration line for label-guided exploration: the moves (UCI) a
        demonstrator played from the current position onwards, both sides. While the
        game follows the line, `demo_actions()` returns the next demonstrated move."""
        self.demo_line = list(ucis or [])
        self.demo_index = 0
        self.demo_alive = bool(self.demo_line)

    def demo_actions(self) -> set[int]:
        """Action ids the demonstrator would play now: the next move of the line while
        the game still follows it; otherwise, the mating moves if the side to move has
        one (rules); otherwise empty."""
        if self.is_done():
            return set()
        if getattr(self, "demo_alive", False) and self.demo_index < len(self.demo_line):
            ids = self.actions_for_uci([self.demo_line[self.demo_index]])
            if ids:
                return ids
        mates = set(mating_actions_of(self))
        if mates:
            return mates
        return forcing_mate_actions_of(self) if getattr(self, "demo_forcing", True) else set()

    def reset(self, fen: str | None = None, puzzle_moves: int = 0, key_moves=None,
              max_plies: int | None = None, solution=None):
        """Start a new game from the opening, or from `fen` when given.

        OpenSpiel's chess game has no FEN parameter, so a custom start is built
        with `new_initial_state(fen)` and installed with `set_state`.

        `puzzle_moves > 0`: puzzle episode. It ends when the side to move at reset
        (the mover) has made `puzzle_moves` moves, or earlier when the game ends;
        a mate gives the normal result, anything else "puzzle_miss". With
        `puzzle_moves > 1` and `key_moves` (UCI), a first move outside `key_moves`
        ends the episode at once as a miss, so only the forcing line is rewarded.
        `max_plies`: game boards end as a draw after that many plies (AlphaZero
        terminated over-long games as draws, Silver et al. 2017 Methods).
        """
        self.puzzle_moves = int(puzzle_moves)
        self.max_plies = max_plies
        self.plies = 0
        self.mover_moves = 0
        self.failed = False
        self.key_actions = None
        if fen is None:
            time_step = self.env.reset()
        else:
            self.env.reset()
            game = pyspiel.load_game("chess")
            self.env.set_state(game.new_initial_state(fen))
            time_step = self.env.get_time_step()
        self.mover = self.get_current_player()
        if self.puzzle_moves > 1 and key_moves:
            self.key_actions = self.actions_for_uci(key_moves)
        self.set_demo_line(solution if solution else (key_moves if self.puzzle_moves > 1 else None))
        return time_step

    def actions_for_uci(self, ucis) -> set[int]:
        """Map UCI moves to OpenSpiel action ids in the current position.

        OpenSpiel's `action_to_string` is SAN and matches python-chess SAN
        (verified for castling and promotions), so SAN is the bridge."""
        state = self.env.get_state
        player = state.current_player()
        board = chess.Board(state.to_string())
        wanted = set(ucis)
        out = set()
        for a in state.legal_actions():
            san = state.action_to_string(player, a)
            try:
                if board.parse_san(san).uci() in wanted:
                    out.add(a)
            except ValueError as e:
                raise ValueError(f"cannot parse SAN {san!r} in {board.fen()}") from e
        return out

    def step(self, action):
        if not isinstance(action, list):
            action = [action]
        if getattr(self, "demo_alive", False):
            if self.demo_index < len(self.demo_line) and \
                    action[0] in self.actions_for_uci([self.demo_line[self.demo_index]]):
                self.demo_index += 1
            else:
                self.demo_alive = False
        if self.get_current_player() == self.mover:
            self.mover_moves += 1
            if self.mover_moves == 1 and self.key_actions is not None and action[0] not in self.key_actions:
                self.failed = True
        self.plies += 1
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

    def is_terminal(self):
        return self.env.get_time_step().last()

    @property
    def is_puzzle(self) -> bool:
        return self.puzzle_moves > 0

    def is_done(self):
        if self.is_terminal():
            return True
        if self.is_puzzle:
            return self.failed or self.mover_moves >= self.puzzle_moves
        return self.max_plies is not None and self.plies >= self.max_plies

    def game_result(self):
        """Return "white_win" | "black_win" | "draw" | "puzzle_miss", or None if not over.
        A game cut by `max_plies` is a draw; a puzzle episode without a mate is a miss."""
        if not self.is_done():
            return None
        if not self.is_terminal():
            return "puzzle_miss" if self.is_puzzle else "draw"
        rewards = self.env.get_time_step().rewards
        if rewards[1] > 0:
            return "white_win"
        elif rewards[0] > 0:
            return "black_win"
        return "draw"
