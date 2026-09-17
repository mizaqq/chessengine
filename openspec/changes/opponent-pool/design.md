## Context

`_collect_rollout` already routes each ply to `models[player]`, and with the shared
network both entries are the learner. GAE and bootstrap are computed per colour from
whatever `old_value` the rollout stored; `_gather_side` selects a colour's plies for
the batch. The vector env auto-resets a finished board inside `step` and reports the
finished boards in `info["game_results"]`. Board roles run puzzle, finishing,
mid-game, opening; `midgame_boards` is a count and opening is the remainder.

## Goals / Non-Goals

**Goals:** frozen opponents on some game boards; learner-only batches; pool snapshots;
prior score in evaluation; 16-board layout as the default.

**Non-Goals:** NFSP's average policy; opponents on puzzle or finishing boards (those
train the attack, the defender stays the learner); league matchmaking by strength;
changes to rewards or to the update.

## Decisions

- **Pool logic lives in the training loop, not the env.** `OpponentPool` (new
  `src/training/opponent_pool.py`) holds the prior and a deque of snapshots and
  answers `sample()`; a small `PoolBoards` helper keeps per-board (opponent name,
  network, learner colour) and redraws on the boards that `info["game_results"]`
  reports done. The env stays colour-blind, as it is today.
- **Routing.** In the rollout, for each pool board where `player != learner_colour`,
  the ply is taken from the opponent network. Boards sharing an opponent object are
  batched into one forward pass. `old_value` for opponent plies is filled with the
  opponent's value, `old_log_prob` with its log-prob; both are dropped later. The
  `StepRecord` gains `learner: BoolTensor` (all True when the pool is off).
- **Gather.** `_gather_side` adds `& step.learner` to `is_mine`. The GAE for a colour
  still runs over all boards; its output on opponent plies is discarded. The bootstrap
  needs no change: with the shared network `models[opponent_id]` is the learner's own
  network, so the negated value is already the learner's estimate.
- **Snapshots.** After the update at every `opponent_snapshot_every`, the loop deep-
  copies the learner into a frozen eval-mode model and pushes it; the deque drops the
  oldest beyond `opponent_pool_size`. Memory: five small networks, negligible.
- **Prior score in evaluation.** `src/eval/matches.py` gets `play_match(a, b, games,
  seed)` returning counts and score; `scripts/eval_matrix.py` switches to it; the
  eval hook adds `prior_score`, `prior_wins`, `prior_losses` when `opponent_prior`
  is set (20 games per colour, about 35 s).
- **Layout default.** `num_envs: 16`, `puzzle_boards: 4`, `finish_boards: 2`,
  `midgame_boards: 5` (opening 5), `opponent_pool_boards: 5`. Per update 1,024 plies,
  of which 640 whole-game; opponent plies leave roughly 160 of them, so the learner's
  batch is about 860 plies, still above today's 768. Cost per update rises about a
  third (0.27 min at 12 boards → about 0.35 min).

## Hyperparameters

- `opponent_pool_boards` (5): how many game boards face the pool. 0 = today's
  self-play; all game boards = the learner never meets its current self, which
  Bansal's uniform-history setting effectively does; mid values keep both signals.
- `opponent_pool_size` (4): snapshots kept. 0 = prior only, no history; large =
  approaches uniform-over-history, more diversity, older and weaker opponents.
- `opponent_snapshot_every` (50): staleness of the newest snapshot. Very small =
  the pool is nearly the current self (back to plain self-play); very large = the
  pool never catches up with the learner and it can overfit to beating old copies.
- `opponent_prior` (path): the fixed anchor member. Unset = pool of snapshots only,
  which cannot oppose drift that all snapshots share.
- `prior_eval_games` (20 per colour): more games, less noise, longer evaluation
  (0.8 s a game).

## Risks / Trade-offs

- Opponent overfitting (Bansal §5.5.2): the learner learns the pool's habits. Guard:
  the end matrix against members not in the pool (CLEAN_GUIDE).
- A two-policy cycle if the snapshots are too similar; the prior anchors one end.
- Fewer learner plies per game board than in self-play (half), compensated by the
  larger layout.
- The prior's BatchNorm statistics are frozen with it; the prior plays exactly as it
  did in the matrix.
- Learning-rate schedule: the default halves the rate every 100 updates down to
  3e-5, so a long run trains at 3e-5 from update 200; unchanged here, noted for the
  reading.

## Long run (named exception)

Owner 2026-09-17: the next test run is long so the evaluation happens in the
morning. Plan: 800 updates from M34 (seed 42), about 6.5 h with evaluations, then
games, finishes and the win matrix. Standard budget resumes afterwards.
