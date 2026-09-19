## 1. Rules helper

- [ ] 1.1 Move `capture_gain` / `best_gain` into `src/envs/open_spiel_env.py`, add `free_capture_actions_of(env, threshold)` and `OpenSpielEnv.punish_actions()` (mates ∪ free captures); `scripts/blunder_audit.py` imports the shared helpers. Verify: `tests/scripts/test_blunder_audit.py` still passes and a new `tests/envs/test_punish_actions.py` covers hung queen, recapture cancels, threshold.

## 2. Rollout and config

- [ ] 2.1 Config `punish_epsilon` (default 0, in [0,1)), `punish_threshold` (default 3, positive int), `punish_boards` (default `all`); `resolve_punish` in `src/entrypoints/train.py` with validation tests in `tests/entrypoints/test_config.py`.
- [ ] 2.2 Rollout: per-row (demo set, epsilon) choice — label demo with `guide_epsilon` first, else punishing moves with `punish_epsilon`; `guided_probs` with per-row epsilon; component sampling sets the teacher-touched flag; `metrics.add_punish(count, picks)` and summary keys `punish_count`, `punish_picks`. Verify: `tests/model/test_guided.py` additions — hung-queen row has behaviour prob 0.51 and stored log-prob log(0.51); a row with a human-line demo uses `guide_epsilon`; pool-opponent rows untouched; `punish_epsilon 0` reproduces today's rollout bit for bit under a fixed seed.
- [ ] 2.3 Vector env: `punish_actions(boards)` per board like `demo_actions`. Verify: unit test on a two-board env.

## 3. Experiment

- [ ] 3.1 Driver `experiments/punish-gifts/run_punish.sh`: 120 updates from the chosen checkpoint, `punish_epsilon 0.5`, then games, finishes, blunder audit (self / SL), win matrix vs the start checkpoint. Verify: run.json, audit json, matrix json committed.
- [ ] 3.2 Write `experiments/punish-gifts/outcome.md`: gift rates and punish rates before/after, puzzles, own-game m1, entropy, verdict.

## 4. Comprehension

- [ ] 4.1 Comprehension check with owner (CLAUDE.md "Trace by hand"): given a small position and `punish_epsilon` 0.5, work out the behaviour probability and stored log-probability of the punishing move, and say what the technique curriculum sees if that ply wins the game; then run `guided_probs` on the same numbers and compare.
