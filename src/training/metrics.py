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
        self._entropy_game_sum = 0.0
        self._entropy_game_count = 0
        self._entropy_puzzle_sum = 0.0
        self._entropy_puzzle_count = 0
        self._terminal_return_sum = 0.0
        self._terminal_return_count = 0
        self.puzzle_attempts = 0
        self.puzzle_solved = 0
        self._puzzle_by_depth = {}   # depth -> [attempts, solved]
        self.finish_attempts = 0
        self.finish_success = 0
        self.explore_count = 0
        self.explore_offprior = 0
        self._finish_by_depth = {}   # depth -> [attempts, successes]
        self._update_sums = {"policy_loss": 0.0, "value_loss": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0}
        self._update_counts = {k: 0 for k in self._update_sums}

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

    def add_puzzle_result(self, solved: bool, depth: int = 1):
        """Record a finished puzzle episode (kept out of the game rates); `depth` is
        the puzzle's mate-in, reported as `puzzle_solved_rate_m<depth>`."""
        self.puzzle_attempts += 1
        self.puzzle_solved += int(solved)
        entry = self._puzzle_by_depth.setdefault(int(depth), [0, 0])
        entry[0] += 1
        entry[1] += int(solved)

    def add_finish_result(self, depth: int, success: bool):
        """Record a finished finishing-board episode (kept out of the game rates);
        `depth` is the rewind in plies, reported as `finish_success_rate_d<depth>`."""
        self.finish_attempts += 1
        self.finish_success += int(success)
        entry = self._finish_by_depth.setdefault(int(depth), [0, 0])
        entry[0] += 1
        entry[1] += int(success)

    def add_explore(self, count: int, offprior: int):
        """Moves drawn on noisy boards and how many of them the network itself gave
        less than 5% (picks against the prior)."""
        self.explore_count += int(count)
        self.explore_offprior += int(offprior)

    def set_finish_depth(self, depth):
        """Current curriculum depth (state, not a window count)."""
        self.finish_depth = depth

    def add_terminal_return(self, white_view_return: float):
        """Record the unshaped terminal reward (white view) of a finished game."""
        self._terminal_return_sum += white_view_return
        self._terminal_return_count += 1

    def add_entropy(self, raw: float, normalized: float, game=None, puzzle=None):
        """`game` / `puzzle`: normalized entropy on opening boards / puzzle boards."""
        self._entropy_raw_sum += raw
        self._entropy_norm_sum += normalized
        self._entropy_count += 1
        if game is not None:
            self._entropy_game_sum += game
            self._entropy_game_count += 1
        if puzzle is not None:
            self._entropy_puzzle_sum += puzzle
            self._entropy_puzzle_count += 1

    def add_update_stats(self, policy_loss=None, value_loss=None, clip_fraction=None, approx_kl=None):
        """Per-update losses and PPO diagnostics; None values are skipped (a2c has no
        clip fraction) and report as null."""
        for key, val in (("policy_loss", policy_loss), ("value_loss", value_loss),
                         ("clip_fraction", clip_fraction), ("approx_kl", approx_kl)):
            if val is not None:
                self._update_sums[key] += val
                self._update_counts[key] += 1

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
            "mean_entropy_normalized_game": (
                self._entropy_game_sum / self._entropy_game_count if self._entropy_game_count else None
            ),
            "mean_entropy_normalized_puzzle": (
                self._entropy_puzzle_sum / self._entropy_puzzle_count if self._entropy_puzzle_count else None
            ),
            "puzzle_attempts": self.puzzle_attempts,
            "puzzle_solved_rate": (
                self.puzzle_solved / self.puzzle_attempts if self.puzzle_attempts > 0 else None
            ),
            **{
                f"mean_{k}": (self._update_sums[k] / self._update_counts[k] if self._update_counts[k] else None)
                for k in self._update_sums
            },
            **{f"puzzle_solved_rate_m{d}": (v[1] / v[0]) for d, v in sorted(self._puzzle_by_depth.items())},
            **{f"puzzle_attempts_m{d}": v[0] for d, v in sorted(self._puzzle_by_depth.items())},
            "explore_moves": self.explore_count,
            "explore_offprior_share": (self.explore_offprior / self.explore_count if self.explore_count else None),
            "finish_attempts": self.finish_attempts,
            "finish_success_rate": (self.finish_success / self.finish_attempts if self.finish_attempts else None),
            "finish_depth": getattr(self, "finish_depth", None),
            **{f"finish_success_rate_d{d}": (v[1] / v[0]) for d, v in sorted(self._finish_by_depth.items())},
            **{f"finish_attempts_d{d}": v[0] for d, v in sorted(self._finish_by_depth.items())},
            "mean_terminal_return": (
                self._terminal_return_sum / self._terminal_return_count
                if self._terminal_return_count > 0
                else None
            ),
        }
        self._reset_all()
        return summary
