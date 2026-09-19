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
        self.guide_count = 0
        self.guide_demo_picks = 0
        self.punish_count = 0
        self.punish_picks = 0
        self._finish_by_depth = {}   # depth -> [attempts, successes]
        self._pool_by_name = {}      # opponent name -> [games, points]
        self._technique_by_label = {}  # material label -> [attempts, successes]
        self._sil_sums = {}; self._sil_count = 0; self.sil_buffer_size = None
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

    def add_technique_result(self, label: str, success: bool, clean: bool = True):
        """Record a finished technique board (generated bare-king start) under its
        material label; kept out of the game and finishing rates. `clean`: the
        demonstrator never fired in the episode (the curriculum counts only these)."""
        entry = self._technique_by_label.setdefault(str(label), [0, 0, 0, 0])
        entry[0] += 1
        entry[1] += int(success)
        if clean:
            entry[2] += 1
            entry[3] += int(success)

    def add_pool_result(self, name: str, score: float):
        """Record a finished pool-board game: learner's score (win 1, draw 0.5, loss 0)
        against the pool member `name`; kept out of the self-play rates."""
        entry = self._pool_by_name.setdefault(str(name), [0, 0.0])
        entry[0] += 1
        entry[1] += float(score)

    def add_explore(self, count: int, offprior: int):
        """Moves drawn on noisy boards and how many of them the network itself gave
        less than 5% (picks against the prior)."""
        self.explore_count += int(count)
        self.explore_offprior += int(offprior)

    def add_guide(self, count: int, demo_picks: int):
        """Moves drawn on guided boards with a demonstration available, and how many
        of them were the demonstrated move."""
        self.guide_count += int(count)
        self.guide_demo_picks += int(demo_picks)

    def add_punish(self, count: int, picks: int):
        """Learner plies on punish boards with a punishing move available, and how
        many of them the punisher played (change punish-gifts)."""
        self.punish_count += int(count)
        self.punish_picks += int(picks)

    def add_sil_stats(self, buffer_size: int, stats):
        """Self-imitation diagnostics for one update (`stats` None = nothing drawn)."""
        self.sil_buffer_size = int(buffer_size)
        if stats:
            for k in ("sil_policy_loss", "sil_value_loss", "sil_valid_share", "sil_positive_share"):
                self._sil_sums[k] = self._sil_sums.get(k, 0.0) + float(stats[k])
            self._sil_count += 1

    def set_technique_levels(self, levels):
        """Current technique curriculum level per material set (state, not a window count)."""
        self.technique_dials = {f"technique_level_{k}": v for k, v in levels.items()}

    def set_technique_dials(self, dials):
        """Curriculum state keys already named (technique_level_<set> or the good-starts counts)."""
        self.technique_dials = dict(dials)

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
            "guide_moves": self.guide_count,
            "guide_demo_share": (self.guide_demo_picks / self.guide_count if self.guide_count else None),
            "punish_moves": self.punish_count,
            "punish_picks": self.punish_picks,
            **dict(sorted(getattr(self, "technique_dials", {}).items())),
            **{f"technique_attempts_{k}": v[0] for k, v in sorted(self._technique_by_label.items())},
            **{f"technique_success_rate_{k}": v[1] / v[0] for k, v in sorted(self._technique_by_label.items())},
            **{f"technique_clean_attempts_{k}": v[2] for k, v in sorted(self._technique_by_label.items())},
            **{f"technique_clean_success_rate_{k}": (v[3] / v[2] if v[2] else None)
               for k, v in sorted(self._technique_by_label.items())},
            **({"sil_buffer_size": self.sil_buffer_size,
                **{k: (self._sil_sums.get(k, 0.0) / self._sil_count if self._sil_count else None)
                   for k in ("sil_policy_loss", "sil_value_loss", "sil_valid_share", "sil_positive_share")}}
               if self.sil_buffer_size is not None else {}),
            "pool_games": sum(v[0] for v in self._pool_by_name.values()),
            "pool_score": (sum(v[1] for v in self._pool_by_name.values()) / sum(v[0] for v in self._pool_by_name.values())
                           if self._pool_by_name else None),
            **{f"pool_score_{n}": v[1] / v[0] for n, v in sorted(self._pool_by_name.items())},
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
