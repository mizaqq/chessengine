## 1. Puzzle data

- [x] 1.1 Add `zstandard` as a dev dependency (PyPI index workaround) and verify `uv run python -c "import zstandard"` succeeds
- [x] 1.2 Write `scripts/prepare_puzzles.py`: download the Lichess dump to gitignored `data/raw/`, keep `mateIn1` rows, apply the first move, validate every listed move mates, shuffle with seed 0, write `data/puzzles/mate_in_1_train.csv` (20,000) and `mate_in_1_eval.csv` (2,000); verify row counts and disjoint ids with a test on the written files
- [x] 1.3 Add a loader `load_puzzles(path)` in `src/envs/start_positions.py` and a test that a 3-row fixture loads with fen, moves and rating

## 2. Start positions in environments

- [x] 2.1 `OpenSpielEnv.reset(fen=None)`: build the state from FEN via `new_initial_state` and `set_state`; test the rook-mate scenario (white to move, material 5, exactly one terminal winning move)
- [x] 2.2 `StartPositionSampler(puzzles, fraction, seed)` with `sample()`; tests for fraction 0, fraction 1 and same-seed determinism
- [x] 2.3 Sync vector env accepts a sampler, uses it on reset and auto-reset; test auto-reset with fraction 1 yields a non-opening position
- [x] 2.4 Async vector env passes a per-worker sampler; same tests as 2.3 through the async API
- [x] 2.5 Config: `puzzle_fraction` (default 0.0, range check), `puzzle_train_file` required when fraction > 0; tests for the two error scenarios and the default

## 3. Puzzle evaluation

- [x] 3.1 `src/eval/puzzles.py`: `evaluate_mate_in_one(white_model, black_model, puzzles)` returning top-1 accuracy, mean mating-move probability and count; tests with a stub perfect policy and a uniform policy on the rook-mate position (0.05)
- [x] 3.2 Training loop: optional `eval_fn` and `eval_interval` in `run_chess_training`; metrics merged into the log entry at each interval and at the end; test with a counting stub over 120 updates of a tiny run (calls at 50, 100, 120)
- [x] 3.3 Wire `puzzle_eval_file` and `eval_interval` in `train.py` and `train_default.yaml`; smoke test with a 2-update run and a 3-row eval file

## 3b. Revision after run 1: one-move puzzle episodes on fixed boards

- [x] 3b.1 `OpenSpielEnv.reset(fen, one_move=True)`: episode ends after one ply; `game_result()` returns `puzzle_miss` when not terminal; tests for mate found, mate missed, opening unaffected
- [x] 3b.2 Sampler always returns a puzzle; sync and async vector envs take `num_puzzle_envs`, first boards are puzzle boards with auto-reset to a new puzzle; `info["puzzle_boards"]` marks their finished episodes; tests for the two-of-four and auto-reset scenarios
- [x] 3b.3 Training: `puzzle_miss` terminal rewards (mover gets `puzzle_miss_reward`, other 0); metrics `puzzle_attempts` / `puzzle_solved_rate`, opening-only game rates; tests for the reward mapping and the split window
- [x] 3b.4 Config: `num_puzzle_envs = round(puzzle_fraction * num_envs)`, `puzzle_miss_reward` default 0; smoke test with 1 puzzle board of 2

## 4. Measure

- [x] 4.0 Run 1 (mixing by reset): puzzle top-1 flat 0.04-0.08 vs 0.074 baseline, 423 games / 90k plies, games eval 2/157; recorded in `experiments/mate-in-one-curriculum/frac05/`

- [x] 4.1 Evaluate the scale-0.2 checkpoints on the held-out set; record the baseline numbers in `experiments/mate-in-one-curriculum/baseline.json`
- [ ] 4.2 Run 2: 500 updates, seed 0, scale 0.2, `puzzle_fraction` 0.5, saving to `experiments/mate-in-one-curriculum/frac05-onemove/`; record `run.json` with the puzzle curve, draw rate and wall-clock
- [ ] 4.3 Play 40 sampled games from the opening with the new checkpoints; record in-game mate-in-one rate and draw rate next to the 3.7% / 76% baseline; compare with the predictions in proposal.md in conversation

## 5. Record

- [x] 5.1 README config table and roadmap 04/06 updated; `learning/RESOURCES.md` gains Florensa et al. 2017, Salimans & Chen 2018, Lichess puzzle DB
- [ ] 5.2 Comprehension check with owner (CLAUDE.md "Trace by hand" on the sampler and returns for a one-move puzzle episode, and "Diagnose from symptoms" on the puzzle curve), code pasted inline; outcome goes to `learning/records/`
