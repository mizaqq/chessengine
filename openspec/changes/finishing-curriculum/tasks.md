## 1. Data

- [x] 1.1 `scripts/prepare_finishes.py`: stream the Lichess month, keep checkmate games with both Elo >= 1500, write `data/finishes/finish_train.csv` / `finish_eval.csv` (`game_id,anchor_fen,moves,winner`, anchor at most 40 plies before the end, split by game). Verify: unit tests for the record builder (short game, resignation rejected) in `tests/scripts/test_prepare_finishes.py`; run for 20,000 train / 2,000 held-out.
- [x] 1.2 Re-harvest own-game mate-in-one positions from `experiments/human-pretraining/ppo_from_sl` with `scripts/harvest_positions.py` into `data/puzzles/selfplay_mate_in_1_{train,eval}.csv`. Verify: files rewritten, eval count 300, positions carry the new checkpoint's game phase (mean ply > 20).

## 2. Environment

- [x] 2.1 `src/envs/start_positions.py`: `FinishRecord`, `load_finishes`, `FinishCurriculum(records, depth_start, depth_max, advance_rate, window, cap_margin, seed)` with `sample() -> FinishStart(fen, depth, cap, winner)`, `report(depth, success)`, `current_depth`, `with_seed`. Verify: tests for depth clipping to the record, even depths only, advance at 61/100, no advance at 59/100, ceiling.
- [x] 2.2 Sync vector env: finishing boards `[P, P+F)`, per-board cap, `info["finish_boards"|"finish_depth"|"finish_success"]`, `sampler.report` on end; mid-game and opening boards shift after them. Async env raises on `num_finish_envs > 0`. Verify: `tests/envs/test_start_from_fen.py` depth-0 win, cap draw as failure, layout indices.

## 3. Training loop, metrics, config

- [x] 3.1 `_collect_rollout` routes finishing boards to `metrics.add_finish_result`; `MetricsAggregator` reports `finish_attempts`, `finish_success_rate`, `finish_success_rate_d<k>`, `finish_depth`. Verify: `tests/training/test_metrics_aggregation.py` summary scenario from the spec.
- [x] 3.2 `src/entrypoints/train.py`: `resolve_finish_boards(config, num_puzzle_envs) -> (FinishCurriculum|None, n)`, validation of the new keys and the board-count limit, wiring into `_create_envs`; `train_default.yaml` gets the 4 / 4 / 2 / 2 layout and documented keys. Verify: `tests/entrypoints/test_config.py` layout and too-many-boards scenarios; smoke run of 2 updates.

- [x] 3.3 Scheduled layout (owner, 2026-09-17: "at some time in the training we should also add opening positions, e.g. at update 200"): `envs.set_layout`, `layout_schedule` config, roles switch at reset. Verify: env test that a running finishing game still reports as a finish and the board resets as an opening board; config resolver test.

## 4. Evaluation

- [x] 4.1 `src/eval/finishes.py`: `evaluate_finishes(white, black, records, depths, games, cap_margin, oriented, seed) -> {finish_rate_d<k>, finish_games_d<k>}` with lockstep batched play; `resolve_eval_fn` adds it when `finish_eval_file` is set. Verify: `tests/eval/test_finishes.py` perfect-finisher stub gives 1.0 at depth 2, short record skipped.
- [x] 4.2 `scripts/eval_finishes.py <ckpt_dir>` for baselines; run on the SL + PPO 600 checkpoint and store `experiments/finishing-curriculum/baseline_finishes.json`. Verify: file present with three depths.

## 5. Experiment

- [ ] 5.0 Arm FIN2 (owner, 2026-09-17): 4 puzzle / 6 finishing / 2 mid-game, threshold 0.4, 300 updates (named exception to the standard budget), schedule 4 finishing / 2 mid-game / 2 opening from update 200. Verify: run.json, games.json, finishes.json committed; compared with FIN and the baseline in outcome.md.

- [x] 5.1 (arm FIN, flat; see outcome.md) `experiments/finishing-curriculum/run.sh`: 120 updates from the SL + PPO 600 checkpoint with the default layout, then `scripts.eval_games` (40 games) and `scripts.eval_finishes`. Verify: `run.json`, `games.json`, `finishes.json` written and committed; results compared with the baselines in chat and recorded in `experiments/finishing-curriculum/outcome.md`.
- [ ] 5.2 `learning/`: RESOURCES entry for Florensa et al. 2017 (fetched 2026-09-17), NOTES section on reverse curriculum and the finishing diagnosis, session file. Verify: files updated and committed.

## 6. Comprehension

- [ ] 6.1 Comprehension check with owner (CLAUDE.md "Diagnose from symptoms"): with the run's `finish_success_rate_d<k>` and `finish_depth` curves pasted in chat, the owner names whether the curriculum stalled because of the defender or the attacker and which code region to inspect. Recorded in `learning/records/` only if demonstrated.
