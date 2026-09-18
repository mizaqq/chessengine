## 1. Data

- [ ] 1.1 `scripts/prepare_puzzles.py --themes endgame --depth any`: theme filter, depth from `mateIn<N>` (1-5), output prefix `endgame_mate`; exclude ids present in the existing held-out files. Tests: kept / rejected scenarios. Run: 20k train / 2k eval, committed.

## 2. Technique starts

- [x] 2.1 `FinishStart.config`; `src/envs/technique.py`: `generate_position(set, rng)`, `TechniqueSampler(sets, caps, seed)` with `sample()`, `report()` no-op, `current_depth = 0`. Tests: queen-start scenario (material, side to move, no mate in one, cap, label), reproducible scenario, rejection of check on the strong king.
- [x] 2.2 Vector env passes `finish_config` in info; rollout routes labelled results to `metrics.add_technique_result`; metrics keys `technique_attempts_<label>`, `technique_success_rate_<label>`; finishing counters untouched. Tests: labelled-start-counted-apart scenario, cap-reached scenario on a `Q` board.
- [x] 2.3 `src/eval/technique.py`: `evaluate_technique(...)` with fixed held-out starts; test: fixed-positions scenario and key names.
- [x] 2.4 Config: `finish_source`, `technique_sets`, `technique_caps`, `technique_eval_games`; `resolve_finish_boards` builds the technique sampler when `finish_source: technique`; eval hook adds the technique evaluation; default mix five sources at 0.2, `lichess_end` eval set. Tests: resolver and mix.

## 3. Run and record

- [ ] 3.1 Baselines on POOL_LONG: technique held-out per set, `lichess_end_top1`. Arm TECH: 120 updates from POOL_LONG, default config; games, finishes, probe; `experiments/endgame-technique/`.
- [ ] 3.2 outcome.md against the dials; NOTES section on forced-win curricula; RESOURCES note (no new source; tablebases deferred); learning record if understanding is demonstrated.

## 4. Comprehension check with owner

- [ ] 4.1 Review the diff (CLAUDE.md): with the generator and the cap code pasted, answer what happens on a `Q` board when the network shuffles the queen for 40 plies, and why a start with a mate in one available is rejected.
