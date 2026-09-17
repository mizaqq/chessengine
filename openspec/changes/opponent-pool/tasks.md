## 1. Pool and matches

- [x] 1.1 `src/training/opponent_pool.py`: `OpponentPool(prior_model, size, snapshot_every, seed)` with `maybe_snapshot(model, update)`, `sample() -> (name, model)`, `members`; `PoolBoards(board_ids, pool, seed)` with per-board (name, model, learner_colour), `redraw(done_ids)`, `learner_mask(players) -> BoolTensor`. Tests: fill-and-roll scenario, before-first-snapshot scenario, redraw only on done boards.
- [x] 1.2 `src/eval/matches.py`: `play_match(a, b, games_per_colour, seed) -> dict(counts, score, wins, losses)`; `scripts/eval_matrix.py` uses it. Test: even-match scenario.

## 2. Training loop

- [x] 2.1 `StepRecord.learner`; rollout routes pool-board plies to the opponent (grouped by network), fills `learner`; `_gather_side` filters on it; snapshot hook after the update. Tests: learner-is-black scenario (batch has the two black plies only), pool off leaves batch identical to today (same seed), bootstrap sign unchanged.
- [x] 2.2 Metrics: `add_pool_result(name, score)`; summary keys `pool_games`, `pool_score`, `pool_score_<name>`; pool games excluded from self-play rates. Test: two-games-against-prior scenario.
- [x] 2.3 Config and resolution: `num_envs 16`, `puzzle_boards 4`, `finish_boards 2`, `midgame_boards 5`, `opponent_pool_boards 5`, `opponent_pool_size 4`, `opponent_snapshot_every 50`, `opponent_prior`, `prior_eval_games 20`; `resolve_opponent_pool(config)`; eval hook adds `prior_score` when the prior is set. Tests: config resolution, pool boards are the first game boards after puzzle and finishing.

## 3. Run and record

- [ ] 3.1 Smoke run (4 updates) confirms pool keys in the log and step time; long run 800 updates from M34 seed 42 in the background with games, finishes, win matrix (prior, M34, new) appended; `experiments/opponent-pool/`.
- [ ] 3.2 Read the run in the morning against the dials in proposal.md; outcome.md; RESOURCES already cites Bansal 2017; NOTES section on opponent sampling; learning record if understanding is demonstrated.

## 4. Comprehension check with owner

- [ ] 4.1 Trace by hand (CLAUDE.md): given a 4-ply rollout on a pool board with the learner as black, state which plies reach the batch and what the bootstrap value is at the end; then run the test from 2.1 and compare.
