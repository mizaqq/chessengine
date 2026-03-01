from typing import Dict, Any


class MetricsAggregator:
    """Aggregates metrics over training steps and episodes."""

    def __init__(self):
        self.episode_empty_masks = 0
        self.episode_illegal_samples = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0

    def add_step(self, empty_masks: int = 0, illegal_samples: int = 0):
        """Add metrics from a single step."""
        self.episode_empty_masks += empty_masks
        self.episode_illegal_samples += illegal_samples

    def add_terminal_result(
        self, white_win: bool = False, black_win: bool = False, draw: bool = False
    ):
        """Record terminal game result."""
        if white_win:
            self.white_wins += 1
        elif black_win:
            self.black_wins += 1
        elif draw:
            self.draws += 1

    def episode_summary(self) -> Dict[str, Any]:
        """Return summary of episode metrics."""
        return {
            "episode_empty_masks": self.episode_empty_masks,
            "episode_illegal_samples": self.episode_illegal_samples,
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "draws": self.draws,
        }

    def reset_episode_counters(self):
        """Reset per-episode counters."""
        self.episode_empty_masks = 0
        self.episode_illegal_samples = 0
