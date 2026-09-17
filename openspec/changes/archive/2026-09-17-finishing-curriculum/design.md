## Context

See proposal.md - Why. Today the sync vector env (`src/envs/open_spiel_vector_env.py`)
lays boards out as puzzle boards then mid-game boards then opening boards;
`OpenSpielEnv.reset(fen, max_plies=...)` already supports a per-board cap with a
draw result. Rewards are computed from `game_results` per board in
`_collect_rollout`; puzzle boards are routed to puzzle metrics by
`info["puzzle_boards"]`. Samplers are plain objects with `sample()` and
`with_seed()`; the async env copies one per worker.

## Goals / Non-Goals

Goals: finishing boards with an adaptive depth curriculum in the sync env; data
preparation from the existing Lichess month; finishing metrics and a held-out
conversion evaluation; re-harvested own-game evaluation set; one experiment
arm from the SL + PPO 600 checkpoint.

Non-Goals: async support for finishing boards; changes to rewards or the
update; mate-in-three puzzles; any change to the puzzle boards.

## Decisions

- **Records store an anchor FEN plus the tail of moves, not one FEN per depth.**
  One row serves every depth; the sampler replays `len(moves) - 1 - k` moves
  from the anchor with python-chess. Alternative: 21 FENs per game; 20x the
  file for no benefit.
- **Curriculum state lives in the sampler (`FinishCurriculum`).** The vector env
  calls `sampler.report(depth, success)` when a finishing board ends. This keeps
  the env free of policy about depths and makes the curriculum unit-testable.
  Alternative: state in the training loop; would need a back-channel into the
  env for the next reset.
- **Async env rejects finishing boards.** Each worker owns a sampler copy, so
  the rolling success window would be split and depths would advance
  independently. Supporting it needs a result channel back to the parent; not
  worth it while the default is `env_type: sync`.
- **Success = the record's winner wins.** Not "mate within k+1 plies": the
  defender is the same network and may deviate from the human line, so the
  human's mate may vanish while the position stays won. Any win within the cap
  counts; the cap is what forces speed.
- **Finishing results routed like puzzle results.** `info["finish_boards"]`,
  `info["finish_depth"]`, `info["finish_success"]` next to the puzzle keys; the
  rollout skips game-rate accounting for those boards and calls
  `metrics.add_finish_result(depth, success)`. Terminal rewards are unchanged
  (win +2 / loss -2 / draw -0.5 from `game_results`).
- **Held-out evaluation plays the boards in lockstep** on a list of
  `OpenSpielEnv`s with one batched forward per ply, sampled moves, seeded. Same
  shape as the puzzle evaluation, so it plugs into `eval_many` and reports
  `finish_rate_d<k>`.
- **Init checkpoint evaluated before training** with `scripts/eval_finishes.py`
  so every dial has a baseline at update 0.

## Hyperparameters

- `finish_boards` (4): how many boards run finishing episodes. 0 disables the
  feature; num_envs - puzzle boards means no full game is ever played and the
  model never sees an opening.
- `finish_depth_start` (2): first curriculum depth. 0 makes the first rung a
  mate-in-one board that duplicates the puzzle boards; a large value skips the
  easy rungs and the success signal is as sparse as in ordinary games.
- `finish_depth_max` (40): ceiling. Bounded by the 40-ply anchor window; lower
  values stop the curriculum early, so the model never practises long
  conversions.
- `finish_advance_rate` (0.6): success rate needed to add a rung. Near 1 the
  curriculum stalls on the hardest records; near 0 it races ahead before the
  skill is learned and depths become as sparse as ordinary games.
- `finish_window` (100): episodes in the rolling window. Small windows advance
  on noise (with 4 boards and short episodes an update yields 10-30 finishes);
  large windows react slowly.
- `finish_cap_margin` (20): extra plies over k before the draw. 2 demands the
  human's line exactly; very large values remove the pressure and the model
  can shuffle as in normal games.
- `finish_eval_depths` (2, 10, 20), `finish_eval_games` (100): evaluation grid.

## Risks / Trade-offs

- The defender is the same network and starts as the losing side of a human
  game; if it learns to prolong faster than the attacker learns to convert, the
  success rate stalls and the curriculum with it. Visible as flat
  `finish_success_rate_d<k>` with rising game length; the fix would be a frozen
  or greedy defender, out of scope here.
- Finishing boards replace two opening boards; opening play may drift. Watched
  through Lichess m1 / m2 and human agreement.
- Standard budget may miss the knee for larger depths (2026-09-16 lesson);
  stated in the proposal.
