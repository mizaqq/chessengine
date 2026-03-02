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
        }
        self._reset_all()
        return summary
