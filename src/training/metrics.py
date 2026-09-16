from typing import Dict, Any


class MetricsAggregator:
    """Aggregates metrics over training steps with windowed summaries.

    Calling episode_summary() returns stats accumulated since the last
    summary call (or since construction), then resets all counters.
    This gives per-window rates instead of cumulative totals.
    """

    def __init__(self):
        self._reset_all()

    def _reset_all(self):
        self.episode_empty_masks = 0
        self.episode_illegal_samples = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self._entropy_raw_sum = 0.0
        self._entropy_norm_sum = 0.0
        self._entropy_count = 0
        self._terminal_return_sum = 0.0
        self._terminal_return_count = 0
        self.puzzle_attempts = 0
        self.puzzle_solved = 0

    def add_step(self, empty_masks: int = 0, illegal_samples: int = 0):
        self.episode_empty_masks += empty_masks
        self.episode_illegal_samples += illegal_samples

    def add_terminal_result(
        self, white_win: bool = False, black_win: bool = False, draw: bool = False
    ):
        if white_win:
            self.white_wins += 1
        elif black_win:
            self.black_wins += 1
        elif draw:
            self.draws += 1

    def add_puzzle_result(self, solved: bool):
        """Record a finished one-move puzzle episode (kept out of the game rates)."""
        self.puzzle_attempts += 1
        self.puzzle_solved += int(solved)

    def add_terminal_return(self, white_view_return: float):
        """Record the unshaped terminal reward (white view) of a finished game."""
        self._terminal_return_sum += white_view_return
        self._terminal_return_count += 1

    def add_entropy(self, raw: float, normalized: float):
        self._entropy_raw_sum += raw
        self._entropy_norm_sum += normalized
        self._entropy_count += 1

    def episode_summary(self) -> Dict[str, Any]:
        """Return windowed stats and reset counters for the next window."""
        total_games = self.white_wins + self.black_wins + self.draws
        summary = {
            "empty_masks": self.episode_empty_masks,
            "illegal_samples": self.episode_illegal_samples,
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "draws": self.draws,
            "total_games": total_games,
            "white_win_rate": self.white_wins / total_games if total_games > 0 else 0.0,
            "black_win_rate": self.black_wins / total_games if total_games > 0 else 0.0,
            "draw_rate": self.draws / total_games if total_games > 0 else 0.0,
            "mean_entropy_raw": self._entropy_raw_sum / max(self._entropy_count, 1),
            "mean_entropy_normalized": self._entropy_norm_sum / max(self._entropy_count, 1),
            "puzzle_attempts": self.puzzle_attempts,
            "puzzle_solved_rate": (
                self.puzzle_solved / self.puzzle_attempts if self.puzzle_attempts > 0 else None
            ),
            "mean_terminal_return": (
                self._terminal_return_sum / self._terminal_return_count
                if self._terminal_return_count > 0
                else None
            ),
        }
        self._reset_all()
        return summary
